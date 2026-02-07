/*++
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: MIT

Module Name:

    SconvDepthwiseKernelNeon.S

Abstract:

    Optimized AArch64 assembly implementation of the depthwise convolution
    micro-kernel used by the NCHWc single precision path.

    This kernel performs the following optimisations:
      * Produce a fast path for interior output positions where all input
        accesses are guaranteed to be in-bounds and can be loaded with a pair
        of 128-bit loads.
      * When an output position touches padding, only the affected 4-wide
        lanes are checked individually and loaded; others are zeroed.  This
        mirrors the behavior of the C++ helper LoadInputVectorWithBounds.
      * Keep the multiply/accumulate operations tightly scheduled to hide the
        load latency.

    The kernel computes a single output position for a 16 channel block and is
    repeatedly invoked by the high level dispatch code.

--*/

#include "asmmacro.h"

        .text

// Offsets for stack based parameters.  AArch64 passes the first eight
// arguments in registers (x0-x7).  The remaining parameters are read from the
// stack directly.  The layout is defined by the C compiler so use constant
// offsets here.

        .equ    .Ldw_InputBase,             0
        .equ    .Ldw_InputWidth,            8
        .equ    .Ldw_DilatedInputWidth,     16
        .equ    .Ldw_OutputCountLeftPad,    24
        .equ    .Ldw_OutputCount,           32
        .equ    .Ldw_OutputCountRightPad,   40
        .equ    .Ldw_Bias,                  48
        .equ    .Ldw_Flags,                 56

// Prototype
//
// void
// MlasConvDepthwiseFloatKernelNeonAsm(
//     const float* Input,                   // x0
//     const float* Filter,                  // x1
//     float* Output,                        // x2
//     size_t StrideWidth,                   // x3 (bytes)
//     size_t DilationWidth,                 // x4 (bytes)
//     size_t InputStride,                   // x5 (unused)
//     size_t KernelHeight,                  // x6
//     size_t KernelWidth,                   // x7
//     const float* InputBase,               // [sp + 0]
//     size_t InputWidth,                    // [sp + 8]  (bytes)
//     size_t DilatedInputWidth,             // [sp + 16] (bytes)
//     size_t OutputCountLeftPad,            // [sp + 24]
//     size_t OutputCount,                   // [sp + 32]
//     size_t OutputCountRightPad,           // [sp + 40]
//     const float* Bias,                    // [sp + 48]
//     unsigned KernelFlags);                // [sp + 56]
//

        FUNCTION_ENTRY MlasConvDepthwiseFloatKernelNeonAsm

        // Load the stack parameters used in the hot loops.
        ldr     x8, [sp,#.Ldw_InputBase]            // base of valid input row
        ldr     x9, [sp,#.Ldw_InputWidth]           // row width in bytes
        ldr     x10,[sp,#.Ldw_DilatedInputWidth]    // stride between rows
        ldr     x11,[sp,#.Ldw_OutputCountLeftPad]
        ldr     x12,[sp,#.Ldw_OutputCount]
        ldr     x13,[sp,#.Ldw_OutputCountRightPad]
        ldr     x14,[sp,#.Ldw_Bias]
        ldr     w15,[sp,#.Ldw_Flags]

        // Preserve callee-saved registers used by this routine.
        stp     x29,x30,[sp,#-16]!
        stp     x27,x28,[sp,#-16]!
        stp     x25,x26,[sp,#-16]!
        stp     x23,x24,[sp,#-16]!
        stp     x21,x22,[sp,#-16]!
        stp     x19,x20,[sp,#-16]!
        stp     d12,d13,[sp,#-16]!
        stp     d14,d15,[sp,#-16]!        

        // Compute total number of output elements to produce.
        add     x16,x11,x12
        add     x16,x16,x13

        // Load bias vectors when required; otherwise all zeros are used to
        // initialize the accumulators.
        eor     v20.16b, v20.16b, v20.16b
        eor     v21.16b, v21.16b, v21.16b
        eor     v22.16b, v22.16b, v22.16b
        eor     v23.16b, v23.16b, v23.16b
        tbz     w15,#1,1f                        // no bias addition
        ldp     q20,q21,[x14],#32
        ldp     q22,q23,[x14]
1:
        // Constant zero used by ReLU handling.
        eor     v24.16b, v24.16b, v24.16b

        mov     x17,#0                            // output index

// ---------------------------------------------------------------------------
// Loop over output elements.  Each iteration computes one output position for
// 16 channels.
// ---------------------------------------------------------------------------
.Ldw_OutputLoop:
        // Start accumulators from bias or zeros.
        mov     v0.16b, v20.16b
        mov     v1.16b, v21.16b
        mov     v2.16b, v22.16b
        mov     v3.16b, v23.16b

        mov     x20,x1                            // reset filter pointer
        mov     x21,#0                            // kh = 0

        // Base pointer for this output index across the input.
        madd    x19,x17,x3,x0                    // Input + out_idx*StrideWidth

.Ldw_HeightLoop:
        // Compute [row_start,row_end) for the current kernel row.
        madd    x22,x21,x10,x8                    // row_start
        add     x27,x22,x9                        // row_end
        sub     x29,x27,#64                       // row_end - 64 (fast path)
        add     x28,x27,#-16                      // row_end - 16

        // Base address for the first kw element on this row.
        madd    x26,x21,x10,x19                   // input for this row

        mov     x25,x7                             // kw remaining

.Ldw_WidthLoop:
        // Fast path: the 16-lane load fits completely within the row.
        cmp     x26,x22
        b.lo    .Ldw_SlowPath
        cmp     x26,x29
        bhi     .Ldw_SlowPath
        // Load 16 input values for the current position.
        ld1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x26]
        b       .Ldw_DoFma

.Ldw_SlowPath:
        // Zero registers and conditionally load each 4-wide vector when it is
        // entirely within bounds.  This matches the behavior of the C++
        // helper LoadInputVectorWithBounds.
        eor     v16.16b, v16.16b, v16.16b
        eor     v17.16b, v17.16b, v17.16b
        eor     v18.16b, v18.16b, v18.16b
        eor     v19.16b, v19.16b, v19.16b

        mov     x23,x26
        cmp     x23,x22
        b.lt    2f
        cmp     x23,x28
        b.hi    2f
        ldr     q16,[x23]
2:
        add     x23,x26,#16
        cmp     x23,x22
        b.lt    3f
        cmp     x23,x28
        b.hi    3f
        ldr     q17,[x23]
3:
        add     x23,x26,#32
        cmp     x23,x22
        b.lt    4f
        cmp     x23,x28
        b.hi    4f
        ldr     q18,[x23]
4:
        add     x23,x26,#48
        cmp     x23,x22
        b.lt    5f
        cmp     x23,x28
        b.hi    5f
        ldr     q19,[x23]
5:

.Ldw_DoFma:
        // Load filter block and update accumulators.
        ld1     {v12.4s, v13.4s, v14.4s, v15.4s}, [x20], #64
        fmla    v0.4s, v16.4s, v12.4s
        fmla    v1.4s, v17.4s, v13.4s
        fmla    v2.4s, v18.4s, v14.4s
        fmla    v3.4s, v19.4s, v15.4s

        add     x26,x26,x4                        // advance to next kw
        subs    x25,x25,#1
        b.ne    .Ldw_WidthLoop

        add     x21,x21,#1
        cmp     x21,x6
        blt     .Ldw_HeightLoop

        // Compute destination pointer for this output element.
        add     x23,x2,x17,lsl #6                 // 16 floats per output

        // Accumulate existing output when requested.
        tbz     w15,#0,6f
        ld1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x23]
        fadd    v0.4s, v0.4s, v16.4s
        fadd    v1.4s, v1.4s, v17.4s
        fadd    v2.4s, v2.4s, v18.4s
        fadd    v3.4s, v3.4s, v19.4s
6:
        // Optional ReLU activation.
        tbz     w15,#2,8f
        fmax    v0.4s, v0.4s, v24.4s
        fmax    v1.4s, v1.4s, v24.4s
        fmax    v2.4s, v2.4s, v24.4s
        fmax    v3.4s, v3.4s, v24.4s
8:
        st1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x23]

        add     x17,x17,#1
        cmp     x17,x16
        blt     .Ldw_OutputLoop

        ldp     d14,d15,[sp],#16
        ldp     d12,d13,[sp],#16
        ldp     x19,x20,[sp],#16
        ldp     x21,x22,[sp],#16
        ldp     x23,x24,[sp],#16
        ldp     x25,x26,[sp],#16
        ldp     x27,x28,[sp],#16
        ldp     x29,x30,[sp],#16

        ret

        .end
