        ;++
        ;
        ; SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
        ; SPDX-License-Identifier: MIT
        ;
        ; Module Name:
        ;
        ;    SconvPointwiseKernelNeon.asm
        ;
        ; Abstract:
        ;
        ;    A hand written AArch64 vectorised micro-kernel for pointwise (1x1) convolution
        ;    operating on tensors formatted in the NCHWc layout.  The kernel computes
        ;    up to four output positions in parallel which allows the filter weights to
        ;    be re-used across several outputs, greatly reducing memory bandwidth.
        ;
        ;    This is the armasm64 translation of aarch64/SconvPointwiseKernelNeon.S.
        ;    The instruction sequence is identical to that file; only the assembler
        ;    directives and macro syntax differ.
        ;
        ;--

#include "kxarm64.h"

        TEXTAREA

        ; Stack layout for arguments passed on the stack.  The first eight arguments
        ; are in x0-x7, the remaining four are placed on the stack by the caller.

PW_OutputStride EQU 0
PW_OutputCount EQU 8
PW_Bias EQU 16
PW_Flags EQU 24

        ; Kernel flag bits.  Keep these in sync with sconv_nchwc_kernel_neon.h.
PWFlag_Accumulate EQU 1
PWFlag_Bias EQU 2
PWFlag_Relu EQU 4

        ; Size in bytes of one NCHWc block (16 FP32 values).
PW_BlockBytes EQU 64

        ;-------------------------------------------------------------------------
        ;  Helper macros
        ;-------------------------------------------------------------------------

        ; Compute four outputs for a single input channel block.  The accumulators
        ; for the four outputs are held in v16-v31.

        MACRO
        CPK4_FmlaStep

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

        MEND

; CPK4_FmlaLane - one input channel of a four channel group, for four output
; positions, using a lane indexed multiply.
;
; CPK4_FmlaStep above issues one ld1r per output position per input channel:
; sixteen fmla for eight loads, two fmla per load.  The register blocked C++
; implementation instead loads one input vector per output position and
; indexes its four lanes, which is four times fewer input loads and measured
; faster on ResNet-50.
;
; This macro is the assembly form of that.  The caller loads v4-v7 once with
; four input channels for each of the four output positions, then expands this
; macro four times with $lane 0..3.  The filter pointer advances by 64 bytes
; per expansion exactly as before, so the filter access pattern is unchanged.
;
;   x0                    - filter pointer, advanced by 64 bytes per expansion
;   $in0..$in3            - input vectors for output positions 0-3
;   $lane                 - which of the four channels this expansion consumes
;
; The input registers are passed as parameters rather than written literally so
; that the lane operand takes the "$reg..s[$lane]" form already used by
; CPK1_FmlaWithLane below and by the NCHWc kernel.  armasm64 ends a
; substitution at the first '.', so the second one is what reaches the
; assembler.

        MACRO
        CPK4_FmlaLane $in0, $in1, $in2, $in3, $lane

        ldp     q0,q1,[x0],#32
        ldp     q2,q3,[x0],#32
        fmla    v16.4s,v0.4s,$in0..s[$lane]
        fmla    v17.4s,v1.4s,$in0..s[$lane]
        fmla    v18.4s,v2.4s,$in0..s[$lane]
        fmla    v19.4s,v3.4s,$in0..s[$lane]
        fmla    v20.4s,v0.4s,$in1..s[$lane]
        fmla    v21.4s,v1.4s,$in1..s[$lane]
        fmla    v22.4s,v2.4s,$in1..s[$lane]
        fmla    v23.4s,v3.4s,$in1..s[$lane]
        fmla    v24.4s,v0.4s,$in2..s[$lane]
        fmla    v25.4s,v1.4s,$in2..s[$lane]
        fmla    v26.4s,v2.4s,$in2..s[$lane]
        fmla    v27.4s,v3.4s,$in2..s[$lane]
        fmla    v28.4s,v0.4s,$in3..s[$lane]
        fmla    v29.4s,v1.4s,$in3..s[$lane]
        fmla    v30.4s,v2.4s,$in3..s[$lane]
        fmla    v31.4s,v3.4s,$in3..s[$lane]

        MEND

; CPK4_FmlaGroup - four input channels for four output positions.
;
; Loads one vector of four input channels per output position, then consumes
; all four lanes.  Replaces four CPK4_FmlaStep expansions: sixteen ld1r become
; four ldr, while the fmla count and the filter traffic are unchanged.

        MACRO
        CPK4_FmlaGroup

        ldr     q4,[x1],#16
        ldr     q5,[x2],#16
        ldr     q6,[x3],#16
        ldr     q7,[x4],#16
        CPK4_FmlaLane v4, v5, v6, v7, 0
        CPK4_FmlaLane v4, v5, v6, v7, 1
        CPK4_FmlaLane v4, v5, v6, v7, 2
        CPK4_FmlaLane v4, v5, v6, v7, 3

        MEND

        ; Accumulate helper for the single output path.  The input values for the
        ; output are loaded to v0-v3 and each lane is multiplied with a block of
        ; filter coefficients.

        MACRO
        CPK1_FmlaWithLane $Lane, $AReg

        ldp     q4,q5,[x0],#32
        ldp     q6,q7,[x0],#32
        fmla    v16.4s,v4.4s,$AReg..s[$Lane]
        fmla    v17.4s,v5.4s,$AReg..s[$Lane]
        fmla    v18.4s,v6.4s,$AReg..s[$Lane]
        fmla    v19.4s,v7.4s,$AReg..s[$Lane]

        MEND

        ; Compute a single output position.  Results are returned in v16-v19.

        MACRO
        CPK_ComputeOneOutput

        mov     x5,#0
        eor     v16.16b,v16.16b,v16.16b
        eor     v17.16b,v17.16b,v17.16b
        eor     v18.16b,v18.16b,v18.16b
        eor     v19.16b,v19.16b,v19.16b
pw_ic_loop1
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
        blt     pw_ic_loop1

        MEND

        ;-------------------------------------------------------------------------
        ;  Entry point
        ;-------------------------------------------------------------------------

        LEAF_ENTRY MlasConvPointwiseFloatKernelNeonAsm

        ; Load the arguments passed on the stack.
        ldr     x8,[sp,#PW_OutputStride]
        ldr     x9,[sp,#PW_OutputCount]
        ldr     x10,[sp,#PW_Bias]
        ldr     w11,[sp,#PW_Flags]

        ; Spill base arguments so caller-saved registers can be reused freely.
        sub     sp,sp,#96
        stp     x0,x1,[sp,#0]
        stp     x2,x3,[sp,#16]
        stp     x4,x5,[sp,#32]
        stp     x6,x7,[sp,#48]
        str     x10,[sp,#64]               ; bias base
        str     x9,[sp,#72]                ; output count

        mov     x12,#0                     ; current filter set
        cbz     x5,pw_exit                 ; nothing to do

pw_filter_loop
        ; Compute the base pointers for this filter block.
        ldr     x15,[sp,#0]               ; input base
        ldr     x16,[sp,#16]              ; output base
        madd    x16,x12,x8,x16             ; output pointer for this filter
        ldr     x17,[sp,#8]              ; filter base
        ldr     x0,[sp,#56]               ; filter set stride
        madd    x17,x12,x0,x17             ; filter pointer for this filter
        ldr     x10,[sp,#64]              ; bias base
        add     x10,x10,x12,lsl #6         ; bias pointer (if used)
        ldr     x6,[sp,#24]               ; input row stride
        ldr     x7,[sp,#48]               ; input channel stride
        ldr     x13,[sp,#72]              ; output count
        lsr     x14,x13,#2                 ; number of groups of four outputs
        and     x13,x13,#3                 ; remaining outputs
        ldr     x9,[sp,#32]               ; input channel blocks
        cbz     x14,pw_process_remainder

        ; ------------------------------------------------------------------
        ;  Main loop processing 4 outputs at a time.
        ; ------------------------------------------------------------------
        ALIGN 16
pw_groups
        ; Clear accumulators for 4 outputs (16 vectors total).
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

        mov     x5,#0                      ; current input channel block
pw_ic_loop4
        madd    x1,x5,x7,x15               ; input for this block
        add     x2,x1,x6                   ; four rows starting positions
        add     x3,x2,x6
        add     x4,x3,x6
        add     x0,x17,x5,lsl #10          ; filter for this block

        ; The block size is 16, consumed four input channels at a time.  Each
        ; group loads one input vector per output position and indexes its
        ; four lanes, so the sixteen ld1r that CPK4_FmlaStep would issue become
        ; four ldr while the fmla count and filter traffic are unchanged.
        CPK4_FmlaGroup
        CPK4_FmlaGroup
        CPK4_FmlaGroup
        CPK4_FmlaGroup

        add     x5,x5,#1
        cmp     x5,x9
        blt     pw_ic_loop4

        ; -----------------------------------------------------------------
        ; Store the four outputs computed above.  There are several cases to
        ; handle based on accumulation, bias and ReLU flags.
        ; -----------------------------------------------------------------

        ; Test if the kernel should accumulate into the existing output.
        tbz     w11,#0,pw_store_nacc

        ; Accumulation path.  Load bias once as it is re-used for all four
        ; stores when present.
        tbz     w11,#1,pw_acc_out0
        ldp     q4,q5,[x10]
        ldp     q6,q7,[x10,#32]
pw_acc_out0
        ; ---- output 0 ----
        ldp     q0,q1,[x16]
        ldp     q2,q3,[x16,#32]
        tbz     w11,#1,pw_acc_add0
        fadd    v0.4s,v0.4s,v4.4s
        fadd    v1.4s,v1.4s,v5.4s
        fadd    v2.4s,v2.4s,v6.4s
        fadd    v3.4s,v3.4s,v7.4s
pw_acc_add0
        fadd    v16.4s,v16.4s,v0.4s
        fadd    v17.4s,v17.4s,v1.4s
        fadd    v18.4s,v18.4s,v2.4s
        fadd    v19.4s,v19.4s,v3.4s
        tbz     w11,#2,pw_acc_st0
        eor     v0.16b,v0.16b,v0.16b
        fmax    v16.4s,v16.4s,v0.4s
        fmax    v17.4s,v17.4s,v0.4s
        fmax    v18.4s,v18.4s,v0.4s
        fmax    v19.4s,v19.4s,v0.4s
pw_acc_st0
        stp     q16,q17,[x16]
        stp     q18,q19,[x16,#32]

        ; ---- output 1 ----
        add     x0,x16,#PW_BlockBytes
        ldp     q0,q1,[x0]
        ldp     q2,q3,[x0,#32]
        tbz     w11,#1,pw_acc_add1
        fadd    v0.4s,v0.4s,v4.4s
        fadd    v1.4s,v1.4s,v5.4s
        fadd    v2.4s,v2.4s,v6.4s
        fadd    v3.4s,v3.4s,v7.4s
pw_acc_add1
        fadd    v20.4s,v20.4s,v0.4s
        fadd    v21.4s,v21.4s,v1.4s
        fadd    v22.4s,v22.4s,v2.4s
        fadd    v23.4s,v23.4s,v3.4s
        tbz     w11,#2,pw_acc_st1
        eor     v0.16b,v0.16b,v0.16b
        fmax    v20.4s,v20.4s,v0.4s
        fmax    v21.4s,v21.4s,v0.4s
        fmax    v22.4s,v22.4s,v0.4s
        fmax    v23.4s,v23.4s,v0.4s
pw_acc_st1
        stp     q20,q21,[x0]
        stp     q22,q23,[x0,#32]

        ; ---- output 2 ----
        add     x0,x0,#PW_BlockBytes
        ldp     q0,q1,[x0]
        ldp     q2,q3,[x0,#32]
        tbz     w11,#1,pw_acc_add2
        fadd    v0.4s,v0.4s,v4.4s
        fadd    v1.4s,v1.4s,v5.4s
        fadd    v2.4s,v2.4s,v6.4s
        fadd    v3.4s,v3.4s,v7.4s
pw_acc_add2
        fadd    v24.4s,v24.4s,v0.4s
        fadd    v25.4s,v25.4s,v1.4s
        fadd    v26.4s,v26.4s,v2.4s
        fadd    v27.4s,v27.4s,v3.4s
        tbz     w11,#2,pw_acc_st2
        eor     v0.16b,v0.16b,v0.16b
        fmax    v24.4s,v24.4s,v0.4s
        fmax    v25.4s,v25.4s,v0.4s
        fmax    v26.4s,v26.4s,v0.4s
        fmax    v27.4s,v27.4s,v0.4s
pw_acc_st2
        stp     q24,q25,[x0]
        stp     q26,q27,[x0,#32]

        ; ---- output 3 ----
        add     x0,x0,#PW_BlockBytes
        ldp     q0,q1,[x0]
        ldp     q2,q3,[x0,#32]
        tbz     w11,#1,pw_acc_add3
        fadd    v0.4s,v0.4s,v4.4s
        fadd    v1.4s,v1.4s,v5.4s
        fadd    v2.4s,v2.4s,v6.4s
        fadd    v3.4s,v3.4s,v7.4s
pw_acc_add3
        fadd    v28.4s,v28.4s,v0.4s
        fadd    v29.4s,v29.4s,v1.4s
        fadd    v30.4s,v30.4s,v2.4s
        fadd    v31.4s,v31.4s,v3.4s
        tbz     w11,#2,pw_acc_st3
        eor     v0.16b,v0.16b,v0.16b
        fmax    v28.4s,v28.4s,v0.4s
        fmax    v29.4s,v29.4s,v0.4s
        fmax    v30.4s,v30.4s,v0.4s
        fmax    v31.4s,v31.4s,v0.4s
pw_acc_st3
        stp     q28,q29,[x0]
        stp     q30,q31,[x0,#32]
        b       pw_advance_group

        ; Non-accumulating path: add bias directly to the results if requested
pw_store_nacc
        tbz     w11,#1,pw_nacc_relu
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
pw_nacc_relu
        tbz     w11,#2,pw_nacc_store
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
pw_nacc_store
        stp     q16,q17,[x16]
        stp     q18,q19,[x16,#32]
        add     x0,x16,#PW_BlockBytes
        stp     q20,q21,[x0]
        stp     q22,q23,[x0,#32]
        add     x0,x0,#PW_BlockBytes
        stp     q24,q25,[x0]
        stp     q26,q27,[x0,#32]
        add     x0,x0,#PW_BlockBytes
        stp     q28,q29,[x0]
        stp     q30,q31,[x0,#32]

pw_advance_group
        add     x15,x15,x6,lsl #2
        add     x16,x16,#(PW_BlockBytes*4)
        subs    x14,x14,#1
        bne     pw_groups

        ; ------------------------------------------------------------------
        ;  Handle the leftover (0..3) output positions.
        ; ------------------------------------------------------------------
pw_process_remainder
        cbz     x13,pw_after_filter
pw_left_loop
        CPK_ComputeOneOutput

        ; Accumulate?
        tbz     w11,#0,pw_left_noacc
        ldp     q0,q1,[x16]
        ldp     q2,q3,[x16,#32]
        tbz     w11,#1,pw_left_add
        ldp     q4,q5,[x10]
        ldp     q6,q7,[x10,#32]
        fadd    v0.4s,v0.4s,v4.4s
        fadd    v1.4s,v1.4s,v5.4s
        fadd    v2.4s,v2.4s,v6.4s
        fadd    v3.4s,v3.4s,v7.4s
pw_left_add
        fadd    v16.4s,v16.4s,v0.4s
        fadd    v17.4s,v17.4s,v1.4s
        fadd    v18.4s,v18.4s,v2.4s
        fadd    v19.4s,v19.4s,v3.4s
        tbz     w11,#2,pw_left_st
        eor     v0.16b,v0.16b,v0.16b
        fmax    v16.4s,v16.4s,v0.4s
        fmax    v17.4s,v17.4s,v0.4s
        fmax    v18.4s,v18.4s,v0.4s
        fmax    v19.4s,v19.4s,v0.4s
pw_left_st
        stp     q16,q17,[x16]
        stp     q18,q19,[x16,#32]
        b       pw_left_next

pw_left_noacc
        tbz     w11,#1,pw_left_nrelu
        ldp     q4,q5,[x10]
        ldp     q6,q7,[x10,#32]
        fadd    v16.4s,v16.4s,v4.4s
        fadd    v17.4s,v17.4s,v5.4s
        fadd    v18.4s,v18.4s,v6.4s
        fadd    v19.4s,v19.4s,v7.4s
pw_left_nrelu
        tbz     w11,#2,pw_left_nst
        eor     v0.16b,v0.16b,v0.16b
        fmax    v16.4s,v16.4s,v0.4s
        fmax    v17.4s,v17.4s,v0.4s
        fmax    v18.4s,v18.4s,v0.4s
        fmax    v19.4s,v19.4s,v0.4s
pw_left_nst
        stp     q16,q17,[x16]
        stp     q18,q19,[x16,#32]
pw_left_next
        add     x15,x15,x6
        add     x16,x16,#PW_BlockBytes
        subs    x13,x13,#1
        bne     pw_left_loop

pw_after_filter
        add     x12,x12,#1
        ldr     x0,[sp,#40]               ; output channel blocks
        cmp     x12,x0
        blt     pw_filter_loop
pw_exit

        add     sp,sp,#96
        ret

        LEAF_END MlasConvPointwiseFloatKernelNeonAsm

        END
