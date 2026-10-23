;++
; SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
; SPDX-License-Identifier: MIT
; 
; Module Name:
; 
;     SconvNchwcKernelNeon.S
; 
; Abstract:
; 
;     This module implements the single precision NCHWC convolution kernel for
;     AArch64 processors with NEON support.
; 
;--


#include "kxarm64.h"

        TEXTAREA
        ALIGN 4

;
; PROCESS_LANE - Load one 16-float filter row and accumulate it using a
; single input lane.
;

        MACRO
        PROCESS_LANE $offset0, $offset1, $inreg, $lane
        ldp     q16, q17, [x7, #$offset0]
        ldp     q18, q19, [x7, #$offset1]
        fmla    v0.4s, v16.4s, $inreg..s[$lane]
        fmla    v1.4s, v17.4s, $inreg..s[$lane]
        fmla    v2.4s, v18.4s, $inreg..s[$lane]
        fmla    v3.4s, v19.4s, $inreg..s[$lane]
        MEND

;
; PROCESS_LANE_PAD1 - As PROCESS_LANE, but accumulating into the second
; output's registers.  CONV_LOOP_PAD_2OUT bounds-tests its two outputs
; separately, so each is accumulated by its own single-output block rather than
; by a fused two-output block; only the filter load is shared.
;

        MACRO
        PROCESS_LANE_PAD1 $offset0, $offset1, $inreg, $lane
        ldp     q16, q17, [x7, #$offset0]
        ldp     q18, q19, [x7, #$offset1]
        fmla    v20.4s, v16.4s, $inreg..s[$lane]
        fmla    v21.4s, v17.4s, $inreg..s[$lane]
        fmla    v22.4s, v18.4s, $inreg..s[$lane]
        fmla    v23.4s, v19.4s, $inreg..s[$lane]
        MEND

;
; PROCESS_LANE_2OUT - Load one 16-float filter row and accumulate it using
; two input lanes for a dual-output path.
;

        MACRO
        PROCESS_LANE_2OUT $offset0, $offset1, $inreg0, $inreg1, $lane
        ldp     q16, q17, [x7, #$offset0]
        ldp     q18, q19, [x7, #$offset1]
        fmla    v0.4s,  v16.4s, $inreg0..s[$lane]
        fmla    v1.4s,  v17.4s, $inreg0..s[$lane]
        fmla    v2.4s,  v18.4s, $inreg0..s[$lane]
        fmla    v3.4s,  v19.4s, $inreg0..s[$lane]
        fmla    v20.4s, v16.4s, $inreg1..s[$lane]
        fmla    v21.4s, v17.4s, $inreg1..s[$lane]
        fmla    v22.4s, v18.4s, $inreg1..s[$lane]
        fmla    v23.4s, v19.4s, $inreg1..s[$lane]
        MEND

;
; PROCESS_LANE_3OUT_4FILT - Load one filter row from each of FOUR filter sets
; and accumulate all of them against three input lanes.
;
; The kernel's filter loop (Lfilter_*_loop) encloses the whole kernel height
; and width walk, so the input window is re-read once per filter set.  For the
; super resolution shapes that is four times over a 12.8 MB tensor, which does
; not fit in L2, so each extra walk is genuine memory traffic rather than a
; cache hit.  The x64 kernel does not pay this: ComputeBlock in
; amd64/SconvKernelCommon.inc broadcasts an input value once and applies it to
; four filter rows held at rdx, rdx+rsi, rbx and rbx+rsi, so one pass over the
; input produces four filter sets of output.
;
; This macro is the equivalent shape for AArch64.  The same twelve accumulators
; are reinterpreted: instead of one filter set across three outputs, they hold
; four filter sets across three outputs, and the input lanes are read once for
; all four.
;
;   v0-v2    filter set 0, outputs 0-2      v6-v8    filter set 2, outputs 0-2
;   v3-v5    filter set 1, outputs 0-2      v9-v11   filter set 3, outputs 0-2
;
; x7 addresses filter set 0; x17, x19 and x20 address sets 1, 2 and 3.  Only
; one quarter of a filter row is processed per call, so the caller issues this
; four times per lane group to cover all sixteen output channels.
;
; The parameters are named $foff, $ireg0..$ireg2 and $lidx rather than anything
; resembling a register name: armasm64 substitutes a parameter wherever its
; spelling appears, so a parameter called $v0 would also rewrite the literal
; register v0 used below.
;

        MACRO
        PROCESS_LANE_3OUT_4FILT $foff, $ireg0, $ireg1, $ireg2, $lidx
        ldr     q12, [x7,  #$foff]
        ldr     q13, [x9,  #$foff]
        ldr     q14, [x13, #$foff]
        ldr     q15, [x16, #$foff]
        fmla    v0.4s,  v12.4s, $ireg0..s[$lidx]
        fmla    v1.4s,  v12.4s, $ireg1..s[$lidx]
        fmla    v2.4s,  v12.4s, $ireg2..s[$lidx]
        fmla    v3.4s,  v13.4s, $ireg0..s[$lidx]
        fmla    v4.4s,  v13.4s, $ireg1..s[$lidx]
        fmla    v5.4s,  v13.4s, $ireg2..s[$lidx]
        fmla    v6.4s,  v14.4s, $ireg0..s[$lidx]
        fmla    v7.4s,  v14.4s, $ireg1..s[$lidx]
        fmla    v8.4s,  v14.4s, $ireg2..s[$lidx]
        fmla    v9.4s,  v15.4s, $ireg0..s[$lidx]
        fmla    v10.4s, v15.4s, $ireg1..s[$lidx]
        fmla    v11.4s, v15.4s, $ireg2..s[$lidx]
        MEND

;
; PROCESS_LANE_3OUT - Load one 16-float filter row and accumulate it using
; three input lanes for a tri-output path.
;

        MACRO
        PROCESS_LANE_3OUT $offset0, $offset1, $inreg0, $inreg1, $inreg2, $lane
        ldp     q16, q17, [x7, #$offset0]
        ldp     q18, q19, [x7, #$offset1]
        fmla    v0.4s,  v16.4s, $inreg0..s[$lane]
        fmla    v1.4s,  v17.4s, $inreg0..s[$lane]
        fmla    v2.4s,  v18.4s, $inreg0..s[$lane]
        fmla    v3.4s,  v19.4s, $inreg0..s[$lane]
        fmla    v4.4s,  v16.4s, $inreg1..s[$lane]
        fmla    v5.4s,  v17.4s, $inreg1..s[$lane]
        fmla    v6.4s,  v18.4s, $inreg1..s[$lane]
        fmla    v7.4s,  v19.4s, $inreg1..s[$lane]
        fmla    v8.4s,  v16.4s, $inreg2..s[$lane]
        fmla    v9.4s,  v17.4s, $inreg2..s[$lane]
        fmla    v10.4s, v18.4s, $inreg2..s[$lane]
        fmla    v11.4s, v19.4s, $inreg2..s[$lane]
        MEND

;
; PROCESS_LANE_4OUT - Load one 16-float filter row and accumulate it using
; four input lanes for a quad-output path.
;

        MACRO
        PROCESS_LANE_4OUT $offset0, $offset1, $inreg0, $inreg1, $inreg2, $inreg3, $lane
        ldp     q16, q17, [x7, #$offset0]
        ldp     q18, q19, [x7, #$offset1]
        fmla    v0.4s,  v16.4s, $inreg0..s[$lane]
        fmla    v1.4s,  v17.4s, $inreg0..s[$lane]
        fmla    v2.4s,  v18.4s, $inreg0..s[$lane]
        fmla    v3.4s,  v19.4s, $inreg0..s[$lane]
        fmla    v4.4s,  v16.4s, $inreg1..s[$lane]
        fmla    v5.4s,  v17.4s, $inreg1..s[$lane]
        fmla    v6.4s,  v18.4s, $inreg1..s[$lane]
        fmla    v7.4s,  v19.4s, $inreg1..s[$lane]
        fmla    v8.4s,  v16.4s, $inreg2..s[$lane]
        fmla    v9.4s,  v17.4s, $inreg2..s[$lane]
        fmla    v10.4s, v18.4s, $inreg2..s[$lane]
        fmla    v11.4s, v19.4s, $inreg2..s[$lane]
        fmla    v12.4s, v16.4s, $inreg3..s[$lane]
        fmla    v13.4s, v17.4s, $inreg3..s[$lane]
        fmla    v14.4s, v18.4s, $inreg3..s[$lane]
        fmla    v15.4s, v19.4s, $inreg3..s[$lane]
        MEND

;
; CONV_LOOP_PAD - Convolution loop with per-position bounds checks. This
; path is used for the padded output regions on the left and right edges.
;

        MACRO
        CONV_LOOP_PAD $Tag
        mov     x7, x4                       ; Filter pointer
        mov     x8, x1                       ; Input row pointer
        mov     x9, x13                      ; Row start pointer
        mov     x10, x27                     ; KernelHeight counter

        cbz     x10, $Tag.Lnum90
        cbz     x28, $Tag.Lnum90

$Tag.Lnum91
        mov     x12, x8                      ; Input pointer for width
        mov     x16, x28                     ; KernelWidth counter

$Tag.Lnum92
        ; Branch if the input pointer lies outside [row_start, row_start + row_width).
        sub     x11, x12, x9
        cmp     x11, x14
        b.hs    $Tag.Lnum93

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

$Tag.Lnum93
        add     x7, x7, #1024
        add     x12, x12, x23
        subs    x16, x16, #1
        b.ne    $Tag.Lnum92

        add     x9, x9, x15
        add     x8, x8, x15
        subs    x10, x10, #1
        b.ne    $Tag.Lnum91

$Tag.Lnum90
        MEND

;
; CONV_LOOP_PAD_2OUT - Convolution loop with per-position bounds checks that
; computes two adjacent output points per iteration.
;
; The padded regions are small but expensive: CONV_LOOP_PAD computes a single
; output per filter load, whereas the interior loops amortize each load over
; three or four.  For the ResNet shapes the left and right pads together are
; two of the fourteen columns at Out=14x14 and two of seven at Out=7x7, so this
; path carries roughly a sixth of the 3x3 convolution work at a third of the
; interior's efficiency.
;
; The two outputs are adjacent, so a single filter load serves both.  Each
; output keeps its own bounds test because one may read padding while the other
; does not; the filter pointer advance is shared and therefore performed once.
; Accumulators are v0-v3 for output 0 and v20-v23 for output 1, matching
; CONV_LOOP_MID_2OUT so the surrounding store code is unchanged.
;

        MACRO
        CONV_LOOP_PAD_2OUT $Tag
        mov     x7, x4                       ; Filter pointer
        mov     x8, x1                       ; Input row pointer (output 0)
        mov     x9, x13                      ; Row start pointer
        mov     x10, x27                     ; KernelHeight counter

        cbz     x10, $Tag.Lnum140
        cbz     x28, $Tag.Lnum140

$Tag.Lnum141
        mov     x12, x8                      ; Input pointer for width (output 0)
        mov     x16, x28                     ; KernelWidth counter

$Tag.Lnum142
        ; Output 0: skip if the input pointer lies outside the row.
        sub     x11, x12, x9
        cmp     x11, x14
        b.hs    $Tag.Lnum143

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

$Tag.Lnum143
        ; Output 1: one stride further along the row, tested independently.
        ;
        ; Every general register is live here -- x7/x8/x9/x10/x12/x16 carry the
        ; loop state and x17 is the bias base -- so x11 is reused for both the
        ; bounds test and the address.  The comparison consumes x11 before the
        ; address is formed, so the two uses do not overlap.
        add     x11, x12, x22                ; input pointer for output 1
        sub     x11, x11, x9                 ; offset from row start
        cmp     x11, x14
        b.hs    $Tag.Lnum144

        add     x11, x9, x11                 ; rebuild the input pointer
        ldp     q4, q5, [x11, #0]
        ldp     q6, q7, [x11, #32]

        PROCESS_LANE_PAD1 0,   32,  v4, 0
        PROCESS_LANE_PAD1 64,  96,  v4, 1
        PROCESS_LANE_PAD1 128, 160, v4, 2
        PROCESS_LANE_PAD1 192, 224, v4, 3
        PROCESS_LANE_PAD1 256, 288, v5, 0
        PROCESS_LANE_PAD1 320, 352, v5, 1
        PROCESS_LANE_PAD1 384, 416, v5, 2
        PROCESS_LANE_PAD1 448, 480, v5, 3
        PROCESS_LANE_PAD1 512, 544, v6, 0
        PROCESS_LANE_PAD1 576, 608, v6, 1
        PROCESS_LANE_PAD1 640, 672, v6, 2
        PROCESS_LANE_PAD1 704, 736, v6, 3
        PROCESS_LANE_PAD1 768, 800, v7, 0
        PROCESS_LANE_PAD1 832, 864, v7, 1
        PROCESS_LANE_PAD1 896, 928, v7, 2
        PROCESS_LANE_PAD1 960, 992, v7, 3

$Tag.Lnum144
        add     x7, x7, #1024
        add     x12, x12, x23
        subs    x16, x16, #1
        b.ne    $Tag.Lnum142

        add     x9, x9, x15
        add     x8, x8, x15
        subs    x10, x10, #1
        b.ne    $Tag.Lnum141

$Tag.Lnum140
        MEND

;
; CONV_LOOP_MID - Convolution loop without bounds checks. Output positions in
; the middle region are guaranteed to be fully in-bounds.
;

        MACRO
        CONV_LOOP_MID $Tag
        mov     x7, x4                       ; Filter pointer
        mov     x8, x1                       ; Input row pointer
        mov     x10, x27                     ; KernelHeight counter

        cbz     x10, $Tag.Lnum100
        cbz     x28, $Tag.Lnum100

$Tag.Lnum101
        mov     x12, x8                      ; Input pointer for width
        mov     x16, x28                     ; KernelWidth counter

$Tag.Lnum102
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
        b.ne    $Tag.Lnum102

        add     x8, x8, x15
        subs    x10, x10, #1
        b.ne    $Tag.Lnum101

$Tag.Lnum100
        MEND

;
; CONV_LOOP_MID_3OUT - Convolution loop without bounds checks that computes
; three adjacent output points per iteration.
;

        MACRO
        CONV_LOOP_MID_3OUT $Tag
        mov     x7, x4                       ; Filter pointer
        mov     x8, x1                       ; Input row pointer (output 0)
        mov     x9, x27                      ; KernelHeight counter

        cbz     x9, $Tag.Lnum110
        cbz     x28, $Tag.Lnum110

$Tag.Lnum111
        mov     x12, x8                      ; Input pointer for width (output 0)
        mov     x16, x28                     ; KernelWidth counter

$Tag.Lnum112
        add     x11, x12, x22                ; Output 1 input pointer
        add     x10, x12, x22, lsl #1        ; Output 2 input pointer

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
        b.ne    $Tag.Lnum112

        add     x8, x8, x15
        subs    x9, x9, #1
        b.ne    $Tag.Lnum111

$Tag.Lnum110
        MEND

;
; CONV_LOOP_MID_3OUT_4FILT - Convolution loop without bounds checks that
; computes three adjacent output points for FOUR filter sets at once.
;
; This is the AArch64 equivalent of the x64 ComputeBlock structure: the input
; window is walked once and each input lane is applied to four filter sets,
; rather than re-walking the window once per filter set.  For a 3x3 kernel
; with FilterCount 4 that reduces the input traffic from 36 window loads to 9.
;
; Filter set 0 is addressed by x7 and sets 1-3 by x9, x19 and x20, each one
; FilterStride further on.  The caller establishes those before entry; they are
; advanced together by #1024 per kernel width step, exactly as the single-set
; loop advances x7.
;
; Register use differs from CONV_LOOP_MID_3OUT.  x7, x9, x13 and x16 address
; the four filter sets, so the kernel width counter moves to x10 and the height
; counter to x14.  Both are scratch: x10 is only live as an input pointer
; within a single width iteration, and x14 holds InputWidth, which only the
; bounds-checked CONV_LOOP_PAD reads and which the caller restores from the
; stack slot at #264 before the right padded region runs.  The three input
; lane groups are loaded into v16-v19 / v21-v24 / v25-v28, leaving v12-v15 to
; hold the four filter rows.
;

        MACRO
        CONV_LOOP_MID_3OUT_4FILT $Tag
        mov     x7, x4                       ; Filter set 0 pointer
        add     x9, x7, x25                  ; Filter set 1 pointer
        add     x13, x9, x25                 ; Filter set 2 pointer
        add     x16, x13, x25                ; Filter set 3 pointer
        mov     x8, x1                       ; Input row pointer (output 0)
        mov     x14, x27                     ; KernelHeight counter

        cbz     x14, $Tag.Lnum150
        cbz     x28, $Tag.Lnum150

$Tag.Lnum151
        mov     x12, x8                      ; Input pointer for width (output 0)
        mov     x10, x28                     ; KernelWidth counter

$Tag.Lnum152
        add     x11, x12, x22                ; Output 1 input pointer
        ldp     q16, q17, [x12, #0]
        ldp     q18, q19, [x12, #32]
        ldp     q21, q22, [x11, #0]
        ldp     q23, q24, [x11, #32]
        add     x11, x12, x22, lsl #1        ; Output 2 input pointer
        ldp     q25, q26, [x11, #0]
        ldp     q27, q28, [x11, #32]

        PROCESS_LANE_3OUT_4FILT 0,   v16, v21, v25, 0
        PROCESS_LANE_3OUT_4FILT 16,  v16, v21, v25, 1
        PROCESS_LANE_3OUT_4FILT 32,  v16, v21, v25, 2
        PROCESS_LANE_3OUT_4FILT 48,  v16, v21, v25, 3
        PROCESS_LANE_3OUT_4FILT 64,  v17, v22, v26, 0
        PROCESS_LANE_3OUT_4FILT 80,  v17, v22, v26, 1
        PROCESS_LANE_3OUT_4FILT 96,  v17, v22, v26, 2
        PROCESS_LANE_3OUT_4FILT 112, v17, v22, v26, 3
        PROCESS_LANE_3OUT_4FILT 128, v18, v23, v27, 0
        PROCESS_LANE_3OUT_4FILT 144, v18, v23, v27, 1
        PROCESS_LANE_3OUT_4FILT 160, v18, v23, v27, 2
        PROCESS_LANE_3OUT_4FILT 176, v18, v23, v27, 3
        PROCESS_LANE_3OUT_4FILT 192, v19, v24, v28, 0
        PROCESS_LANE_3OUT_4FILT 208, v19, v24, v28, 1
        PROCESS_LANE_3OUT_4FILT 224, v19, v24, v28, 2
        PROCESS_LANE_3OUT_4FILT 240, v19, v24, v28, 3

        add     x7,  x7,  #256
        add     x9,  x9,  #256
        add     x13, x13, #256
        add     x16, x16, #256
        add     x12, x12, x23
        subs    x10, x10, #1
        b.ne    $Tag.Lnum152

        add     x8, x8, x15
        subs    x14, x14, #1
        b.ne    $Tag.Lnum151

$Tag.Lnum150
        MEND

;
; CONV_LOOP_MID_2OUT - Convolution loop without bounds checks that computes
; two adjacent output points per iteration.
;

        MACRO
        CONV_LOOP_MID_2OUT $Tag
        mov     x7, x4                       ; Filter pointer
        mov     x8, x1                       ; Input row pointer (output 0)
        add     x9, x1, x22                  ; Input row pointer (output 1)
        mov     x10, x27                     ; KernelHeight counter

        cbz     x10, $Tag.Lnum120
        cbz     x28, $Tag.Lnum120

$Tag.Lnum121
        mov     x12, x8                      ; Input pointer for width (output 0)
        mov     x11, x9                      ; Input pointer for width (output 1)
        mov     x16, x28                     ; KernelWidth counter

$Tag.Lnum122
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
        b.ne    $Tag.Lnum122

        add     x8, x8, x15
        add     x9, x9, x15
        subs    x10, x10, #1
        b.ne    $Tag.Lnum121

$Tag.Lnum120
        MEND

;
; CONV_LOOP_MID_4OUT - Convolution loop without bounds checks that computes
; four adjacent output points per iteration. Used by both the KernelFlags==0
; fast path and the general flags path to reduce filter load pressure: one
; filter load feeds four outputs instead of three.
;
; $KHReg names the register used as the KernelHeight counter so each caller can
; pick one that does not collide with its own live values.
;

        MACRO
        CONV_LOOP_MID_4OUT $Tag, $KHReg
        mov     x7, x4                       ; Filter pointer
        mov     x8, x1                       ; Input row pointer (output 0)
        mov     $KHReg, x27                  ; KernelHeight counter

        cbz     $KHReg, $Tag.Lnum130
        cbz     x28, $Tag.Lnum130

$Tag.Lnum131
        mov     x12, x8                      ; Input pointer for width (output 0)
        mov     x16, x28                     ; KernelWidth counter

$Tag.Lnum132
        add     x9,  x12, x22                ; Output 1 input pointer
        add     x10, x9,  x22                ; Output 2 input pointer
        add     x11, x10, x22                ; Output 3 input pointer

        ; Load lanes 0..7 for all outputs.
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

        ; Load lanes 8..15 for all outputs.
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
        b.ne    $Tag.Lnum132

        add     x8, x8, x15
        subs    $KHReg, $KHReg, #1
        b.ne    $Tag.Lnum131

$Tag.Lnum130
        MEND

;
; void
; MlasConvNchwcFloatKernelNeonAsm(
;     const float* Input,
;     const float* Filter,
;     float* Output,
;     size_t StrideWidth,
;     size_t DilationWidth,
;     size_t FilterCount,
;     size_t InputStride,
;     size_t FilterStride,
;     size_t OutputStride,
;     size_t KernelHeight,
;     size_t KernelWidth,
;     const float* InputBase,
;     size_t InputWidth,
;     size_t DilatedInputWidth,
;     size_t OutputCountLeftPad,
;     size_t OutputCount,
;     size_t OutputCountRightPad,
;     const float* Bias,
;     unsigned KernelFlags
;     );
;

        NESTED_ENTRY MlasConvNchwcFloatKernelNeonAsm

    ; Preserve the incoming stack pointer to access stack-passed arguments.
    mov     x9, sp

    ; Prologue and callee-saved register spill.
    ; Save callee-saved SIMD registers v8-v15 per AArch64 ABI.
    PROLOG_SAVE_REG_PAIR x29, x30, #-272!
    PROLOG_SAVE_REG_PAIR x19, x20, #16
    PROLOG_SAVE_REG_PAIR x21, x22, #32
    PROLOG_SAVE_REG_PAIR x23, x24, #48
    PROLOG_SAVE_REG_PAIR x25, x26, #64
    PROLOG_SAVE_REG_PAIR x27, x28, #80
    PROLOG_NOP stp     q8, q9, [sp, #96]
    PROLOG_NOP stp     q10, q11, [sp, #128]
    PROLOG_NOP stp     q12, q13, [sp, #160]
    PROLOG_NOP stp     q14, q15, [sp, #192]

    ; Move register arguments into callee-saved registers.
    mov     x19, x0                      ; Input
    mov     x20, x1                      ; Filter
    mov     x21, x2                      ; Output
    mov     x22, x3                      ; StrideWidth (bytes)
    mov     x23, x4                      ; DilationWidth (bytes)
    mov     x24, x5                      ; FilterCount
    mov     x25, x7                      ; FilterStride (bytes)

    ; Load stack arguments using the preserved incoming stack pointer.
    ldr     x10, [x9, #0]                ; OutputStride (bytes)
    ldr     x11, [x9, #8]                ; KernelHeight
    ldr     x12, [x9, #16]               ; KernelWidth
    ldr     x13, [x9, #24]               ; InputBase
    ldr     x14, [x9, #32]               ; InputWidth (bytes)
    ldr     x15, [x9, #40]               ; DilatedInputWidth (bytes)
    ldr     x16, [x9, #48]               ; OutputCountLeftPad
    ldr     x17, [x9, #56]               ; OutputCount
    ldr     x6,  [x9, #72]               ; Bias
    ldr     w8,  [x9, #80]               ; KernelFlags

    ; Early exit when nothing to compute.
    mov     x26, x10                     ; OutputStride (bytes)
    ldr     x10, [x9, #64]               ; OutputCountRightPad
    add     x0, x16, x17
    add     x0, x0, x10                  ; x0 = TotalOutputCount
    cbz     x0, Lepilogue
    cbz     x24, Lepilogue

    ; Spill the output counts so that x16/x17 can be used as scratch.
    str     x16, [sp, #224]
    str     x17, [sp, #232]
    str     x10, [sp, #240]
    str     w8,  [sp, #256]

    mov     x27, x11                     ; KernelHeight
    mov     x28, x12                     ; KernelWidth
    mov     x17, x6                      ; Bias

    ; Set up a zero vector for ReLU.
    movi    v31.4s, #0

    ; x1 = current input base for the output index.
    ; x2 = output offset in bytes for the output index.
    mov     x1, x19
    mov     x2, xzr

    ; Fast path when no post-processing flags are enabled. This removes
    ; repeated flag checks and branches from the steady-state loops.
    ldr     w9, [sp, #256]
    tst     w9, #7
    b.eq    Lkernel_flags0

    ; Process the left padded output region with bounds checks.
    ldr     x0, [sp, #224]
    cbz     x0, Loutput_mid_begin

Loutput_left_loop

    ; Initialize per-filter-set pointers and loop counter.
    mov     x3, x24
    mov     x4, x20
    mov     x5, x21
    mov     x6, x17

Lfilter_left_loop

    ; Clear accumulators.
    eor     v0.16b, v0.16b, v0.16b
    eor     v1.16b, v1.16b, v1.16b
    eor     v2.16b, v2.16b, v2.16b
    eor     v3.16b, v3.16b, v3.16b

    ; Convolution loop with bounds checks.
        CONV_LOOP_PAD lb1

    ; Compute the output pointer for this filter set and output index.
    add     x12, x5, x2

    ; Conditionally accumulate the existing output.
    ldr     w9, [sp, #256]
    tst     w9, #1
    b.eq    Lskip_accumulate
    ldp     q16, q17, [x12, #0]
    ldp     q18, q19, [x12, #32]
    fadd    v0.4s, v0.4s, v16.4s
    fadd    v1.4s, v1.4s, v17.4s
    fadd    v2.4s, v2.4s, v18.4s
    fadd    v3.4s, v3.4s, v19.4s

Lskip_accumulate

    ; Conditionally add bias.
    tst     w9, #2
    b.eq    Lskip_bias
    ldp     q16, q17, [x6, #0]
    ldp     q18, q19, [x6, #32]
    fadd    v0.4s, v0.4s, v16.4s
    fadd    v1.4s, v1.4s, v17.4s
    fadd    v2.4s, v2.4s, v18.4s
    fadd    v3.4s, v3.4s, v19.4s

Lskip_bias

    ; Conditionally apply ReLU.
    tst     w9, #4
    b.eq    Lskip_relu
    fmax    v0.4s, v0.4s, v31.4s
    fmax    v1.4s, v1.4s, v31.4s
    fmax    v2.4s, v2.4s, v31.4s
    fmax    v3.4s, v3.4s, v31.4s

Lskip_relu

    ; Store the result.
    stp     q0, q1, [x12, #0]
    stp     q2, q3, [x12, #32]

    ; Advance filter/output/bias pointers for the next filter set block.
    add     x4, x4, x25
    add     x5, x5, x26
    add     x6, x6, #64
    subs    x3, x3, #1
    b.ne    Lfilter_left_loop

    ; Advance to the next output index.
    add     x1, x1, x22
    add     x2, x2, #64
    subs    x0, x0, #1
    b.ne    Loutput_left_loop

    ; Process the middle output region without bounds checks.
Loutput_mid_begin
    ldr     x0, [sp, #232]
    cbz     x0, Loutput_right_begin

    ; Process four outputs at a time to amortize the filter loads, matching
    ; the KernelFlags == 0 path.  Accumulate/bias/ReLU are applied one output
    ; position at a time below, reusing v16-v19 as scratch, so the wider
    ; compute loop costs no extra registers.
    ;
    ; This matters because ComputeKernelFlags sets ACCUMULATE_OUTPUT for every
    ; input channel block after the first (snchwc.cpp), so a convolution with
    ; 256 input channels reaches this path for 15 of its 16 kernel calls.
    and     x16, x0, #3                  ; Remainder outputs after quads.
    str     x16, [sp, #248]
    lsr     x0, x0, #2                   ; Quad output count.
    cbz     x0, Loutput_mid_triad_begin

    ; The bias pointer is kept in x14 rather than x6 because
    ; CONV_LOOP_MID_4OUT uses x6 as its KernelHeight counter.  x14 holds
    ; InputWidth, which is read only by CONV_LOOP_PAD; that macro never runs in
    ; this bounds-check-free middle region, so x14 is saved once here and
    ; restored once the quad region is finished.
    str     x14, [sp, #264]

Loutput_mid_quad_loop

    ; Initialize per-filter-set pointers and loop counter.
    mov     x3, x24
    mov     x4, x20
    mov     x5, x21
    mov     x14, x17                     ; Bias pointer for this output quad.

Lfilter_mid_quad_loop

    ; Clear accumulators for four output points.
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

    ; Convolution loop without bounds checks computing four outputs.
        CONV_LOOP_MID_4OUT lb12, x6

    ; Compute the output pointers for the four output points.
    add     x12, x5, x2
    add     x11, x12, #64
    add     x10, x12, #128
    add     x9,  x12, #192

    ; KernelFlags is held in w16 here, not w9 as elsewhere, because x9 is the
    ; fourth output pointer in this block.  x16 is scratch inside
    ; CONV_LOOP_MID_4OUT and dead once the macro returns.
    ldr     w16, [sp, #256]

    ; Conditionally accumulate the existing output.  Each output position is
    ; handled separately so that only four scratch registers are needed.
    tst     w16, #1
    b.eq    Lskip_accumulate_quad
    ldp     q16, q17, [x12, #0]
    ldp     q18, q19, [x12, #32]
    fadd    v0.4s, v0.4s, v16.4s
    fadd    v1.4s, v1.4s, v17.4s
    fadd    v2.4s, v2.4s, v18.4s
    fadd    v3.4s, v3.4s, v19.4s
    ldp     q16, q17, [x11, #0]
    ldp     q18, q19, [x11, #32]
    fadd    v4.4s, v4.4s, v16.4s
    fadd    v5.4s, v5.4s, v17.4s
    fadd    v6.4s, v6.4s, v18.4s
    fadd    v7.4s, v7.4s, v19.4s
    ldp     q16, q17, [x10, #0]
    ldp     q18, q19, [x10, #32]
    fadd    v8.4s, v8.4s, v16.4s
    fadd    v9.4s, v9.4s, v17.4s
    fadd    v10.4s, v10.4s, v18.4s
    fadd    v11.4s, v11.4s, v19.4s
    ldp     q16, q17, [x9, #0]
    ldp     q18, q19, [x9, #32]
    fadd    v12.4s, v12.4s, v16.4s
    fadd    v13.4s, v13.4s, v17.4s
    fadd    v14.4s, v14.4s, v18.4s
    fadd    v15.4s, v15.4s, v19.4s

Lskip_accumulate_quad

    ; Conditionally add bias.  The same bias block applies to all four outputs.
    tst     w16, #2
    b.eq    Lskip_bias_quad
    ldp     q16, q17, [x14, #0]
    ldp     q18, q19, [x14, #32]
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
    fadd    v12.4s, v12.4s, v16.4s
    fadd    v13.4s, v13.4s, v17.4s
    fadd    v14.4s, v14.4s, v18.4s
    fadd    v15.4s, v15.4s, v19.4s

Lskip_bias_quad

    ; Conditionally apply ReLU.
    tst     w16, #4
    b.eq    Lskip_relu_quad
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
    fmax    v12.4s, v12.4s, v31.4s
    fmax    v13.4s, v13.4s, v31.4s
    fmax    v14.4s, v14.4s, v31.4s
    fmax    v15.4s, v15.4s, v31.4s

Lskip_relu_quad

    ; Store the results for the four output points.
    stp     q0, q1, [x12, #0]
    stp     q2, q3, [x12, #32]
    stp     q4, q5, [x11, #0]
    stp     q6, q7, [x11, #32]
    stp     q8, q9, [x10, #0]
    stp     q10, q11, [x10, #32]
    stp     q12, q13, [x9, #0]
    stp     q14, q15, [x9, #32]

    ; Advance filter/output/bias pointers for the next filter set block.
    add     x4, x4, x25
    add     x5, x5, x26
    add     x14, x14, #64
    subs    x3, x3, #1
    b.ne    Lfilter_mid_quad_loop

    ; Advance to the next four output indices.
    add     x1, x1, x22, lsl #2
    add     x2, x2, #256
    subs    x0, x0, #1
    b.ne    Loutput_mid_quad_loop

    ; Restore InputWidth for the right padded region's bounds checks.
    ldr     x14, [sp, #264]

    ; Handle the 0..3 outputs left over after the quad loop.
Loutput_mid_triad_begin
    ldr     x0, [sp, #248]
    cbz     x0, Loutput_right_begin
    cmp     x0, #3
    b.ne    Loutput_mid_triad_skip
    ; Exactly three outputs remain: run one triad and no further remainder.
    str     xzr, [sp, #248]
    mov     x0, #1
    b       Loutput_mid_triad_loop

Loutput_mid_triad_skip
    ; Fewer than three remain, so fall through to the pair/single handling.
    b       Loutput_mid_pair_begin

Loutput_mid_triad_loop

    ; Initialize per-filter-set pointers and loop counter.
    mov     x3, x24
    mov     x4, x20
    mov     x5, x21
    mov     x6, x17

    ; When four or more filter sets remain, process four at a time with a
    ; single walk of the input window.  Fewer than four falls through to the
    ; original one-set-at-a-time loop below, which also handles the remainder
    ; left over by this loop.
    cmp     x3, #4
    b.lo    Lfilter_mid_triad_loop

Lfilter_mid_triad_4filt_loop

    ; Clear accumulators.  v0-v2 hold filter set 0 across outputs 0-2, v3-v5
    ; set 1, v6-v8 set 2 and v9-v11 set 3.
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

    ; One walk of the input window feeding four filter sets.
        CONV_LOOP_MID_3OUT_4FILT lb14

    ; Output pointers.  x12 is filter set 0's base for output 0; each further
    ; set is one OutputStride on, and each further output 64 bytes on.
    add     x12, x5, x2
    add     x13, x12, x26
    add     x16, x13, x26
    add     x11, x16, x26

    ldr     w9, [sp, #256]

    ; Conditionally accumulate the existing output.
    tst     w9, #1
    b.eq    Lskip_accumulate_triad_4filt
    ldr     q16, [x12, #0]
    ldr     q17, [x12, #64]
    ldr     q18, [x12, #128]
    ldr     q19, [x13, #0]
    ldr     q20, [x13, #64]
    ldr     q21, [x13, #128]
    fadd    v0.4s, v0.4s, v16.4s
    fadd    v1.4s, v1.4s, v17.4s
    fadd    v2.4s, v2.4s, v18.4s
    fadd    v3.4s, v3.4s, v19.4s
    fadd    v4.4s, v4.4s, v20.4s
    fadd    v5.4s, v5.4s, v21.4s
    ldr     q16, [x16, #0]
    ldr     q17, [x16, #64]
    ldr     q18, [x16, #128]
    ldr     q19, [x21, #0]
    ldr     q20, [x21, #64]
    ldr     q21, [x21, #128]
    fadd    v6.4s,  v6.4s,  v16.4s
    fadd    v7.4s,  v7.4s,  v17.4s
    fadd    v8.4s,  v8.4s,  v18.4s
    fadd    v9.4s,  v9.4s,  v19.4s
    fadd    v10.4s, v10.4s, v20.4s
    fadd    v11.4s, v11.4s, v21.4s

Lskip_accumulate_triad_4filt

    ; Conditionally add bias.  Each filter set has its own bias vector, and
    ; that one vector is shared by all three of the set's output points.
    tst     w9, #2
    b.eq    Lskip_bias_triad_4filt
    ldr     q16, [x6, #0]
    ldr     q17, [x6, #16]
    ldr     q18, [x6, #32]
    ldr     q19, [x6, #48]
    fadd    v0.4s, v0.4s, v16.4s
    fadd    v1.4s, v1.4s, v16.4s
    fadd    v2.4s, v2.4s, v16.4s
    fadd    v3.4s, v3.4s, v17.4s
    fadd    v4.4s, v4.4s, v17.4s
    fadd    v5.4s, v5.4s, v17.4s
    fadd    v6.4s,  v6.4s,  v18.4s
    fadd    v7.4s,  v7.4s,  v18.4s
    fadd    v8.4s,  v8.4s,  v18.4s
    fadd    v9.4s,  v9.4s,  v19.4s
    fadd    v10.4s, v10.4s, v19.4s
    fadd    v11.4s, v11.4s, v19.4s

Lskip_bias_triad_4filt

    ; Conditionally apply ReLU.
    tst     w9, #4
    b.eq    Lskip_relu_triad_4filt
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

Lskip_relu_triad_4filt

    ; Store each filter set's three output points.
    str     q0, [x12, #0]
    str     q1, [x12, #64]
    str     q2, [x12, #128]
    str     q3, [x13, #0]
    str     q4, [x13, #64]
    str     q5, [x13, #128]
    str     q6, [x16, #0]
    str     q7, [x16, #64]
    str     q8, [x16, #128]
    str     q9, [x21, #0]
    str     q10, [x21, #64]
    str     q11, [x21, #128]

    ; Advance filter/output/bias pointers over the four sets just processed.
    add     x4, x4, x25, lsl #2
    add     x5, x5, x26, lsl #2
    add     x6, x6, #64
    sub     x3, x3, #4
    cmp     x3, #4
    b.hs    Lfilter_mid_triad_4filt_loop

    ; Fall through to the single-set loop for any remaining filter sets.
    cbz     x3, Lfilter_mid_triad_done

Lfilter_mid_triad_loop

    ; Clear accumulators for three output points.
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

    ; Convolution loop without bounds checks computing three outputs.
        CONV_LOOP_MID_3OUT lb2

    ; Compute the output pointers for the three output points.
    add     x12, x5, x2
    add     x11, x12, #64
    add     x10, x12, #128

    ; Conditionally accumulate the existing output.
    ldr     w9, [sp, #256]
    tst     w9, #1
    b.eq    Lskip_accumulate_triad
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

Lskip_accumulate_triad

    ; Conditionally add bias.
    tst     w9, #2
    b.eq    Lskip_bias_triad
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

Lskip_bias_triad

    ; Conditionally apply ReLU.
    tst     w9, #4
    b.eq    Lskip_relu_triad
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

Lskip_relu_triad

    ; Store the results for the three output points.
    stp     q0, q1, [x12, #0]
    stp     q2, q3, [x12, #32]
    stp     q4, q5, [x11, #0]
    stp     q6, q7, [x11, #32]
    stp     q8, q9, [x10, #0]
    stp     q10, q11, [x10, #32]

    ; Advance filter/output/bias pointers for the next filter set block.
    add     x4, x4, x25
    add     x5, x5, x26
    add     x6, x6, #64
    subs    x3, x3, #1
    b.ne    Lfilter_mid_triad_loop

Lfilter_mid_triad_done

    ; Advance to the next three output indices.
    add     x1, x1, x22, lsl #1
    add     x1, x1, x22
    add     x2, x2, #192
    subs    x0, x0, #1
    b.ne    Loutput_mid_triad_loop

Loutput_mid_pair_begin
    ldr     x0, [sp, #248]
    and     x16, x0, #1
    str     x16, [sp, #248]
    lsr     x0, x0, #1
    cbz     x0, Loutput_mid_single_begin

Loutput_mid_pair_loop

    ; Initialize per-filter-set pointers and loop counter.
    mov     x3, x24
    mov     x4, x20
    mov     x5, x21
    mov     x6, x17

Lfilter_mid_pair_loop

    ; Clear accumulators for both output points.
    eor     v0.16b, v0.16b, v0.16b
    eor     v1.16b, v1.16b, v1.16b
    eor     v2.16b, v2.16b, v2.16b
    eor     v3.16b, v3.16b, v3.16b
    eor     v20.16b, v20.16b, v20.16b
    eor     v21.16b, v21.16b, v21.16b
    eor     v22.16b, v22.16b, v22.16b
    eor     v23.16b, v23.16b, v23.16b

    ; Convolution loop without bounds checks computing two outputs.
        CONV_LOOP_MID_2OUT lb3

    ; Compute the output pointers for the two output points.
    add     x12, x5, x2
    add     x11, x12, #64

    ; Conditionally accumulate the existing output.
    ldr     w9, [sp, #256]
    tst     w9, #1
    b.eq    Lskip_accumulate_pair
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

Lskip_accumulate_pair

    ; Conditionally add bias.
    tst     w9, #2
    b.eq    Lskip_bias_pair
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

Lskip_bias_pair

    ; Conditionally apply ReLU.
    tst     w9, #4
    b.eq    Lskip_relu_pair
    fmax    v0.4s, v0.4s, v31.4s
    fmax    v1.4s, v1.4s, v31.4s
    fmax    v2.4s, v2.4s, v31.4s
    fmax    v3.4s, v3.4s, v31.4s
    fmax    v20.4s, v20.4s, v31.4s
    fmax    v21.4s, v21.4s, v31.4s
    fmax    v22.4s, v22.4s, v31.4s
    fmax    v23.4s, v23.4s, v31.4s

Lskip_relu_pair

    ; Store the results for the two output points.
    stp     q0, q1, [x12, #0]
    stp     q2, q3, [x12, #32]
    stp     q20, q21, [x11, #0]
    stp     q22, q23, [x11, #32]

    ; Advance filter/output/bias pointers for the next filter set block.
    add     x4, x4, x25
    add     x5, x5, x26
    add     x6, x6, #64
    subs    x3, x3, #1
    b.ne    Lfilter_mid_pair_loop

    ; Advance to the next two output indices.
    add     x1, x1, x22, lsl #1
    add     x2, x2, #128
    subs    x0, x0, #1
    b.ne    Loutput_mid_pair_loop

Loutput_mid_single_begin
    ldr     x0, [sp, #248]
    cbz     x0, Loutput_right_begin

Loutput_mid_loop

    ; Initialize per-filter-set pointers and loop counter.
    mov     x3, x24
    mov     x4, x20
    mov     x5, x21
    mov     x6, x17

Lfilter_mid_loop

    ; Clear accumulators.
    eor     v0.16b, v0.16b, v0.16b
    eor     v1.16b, v1.16b, v1.16b
    eor     v2.16b, v2.16b, v2.16b
    eor     v3.16b, v3.16b, v3.16b

    ; Convolution loop without bounds checks.
        CONV_LOOP_MID lb4

    ; Compute the output pointer for this filter set and output index.
    add     x12, x5, x2

    ; Conditionally accumulate the existing output.
    ldr     w9, [sp, #256]
    tst     w9, #1
    b.eq    Lskip_accumulate_mid
    ldp     q16, q17, [x12, #0]
    ldp     q18, q19, [x12, #32]
    fadd    v0.4s, v0.4s, v16.4s
    fadd    v1.4s, v1.4s, v17.4s
    fadd    v2.4s, v2.4s, v18.4s
    fadd    v3.4s, v3.4s, v19.4s

Lskip_accumulate_mid

    ; Conditionally add bias.
    tst     w9, #2
    b.eq    Lskip_bias_mid
    ldp     q16, q17, [x6, #0]
    ldp     q18, q19, [x6, #32]
    fadd    v0.4s, v0.4s, v16.4s
    fadd    v1.4s, v1.4s, v17.4s
    fadd    v2.4s, v2.4s, v18.4s
    fadd    v3.4s, v3.4s, v19.4s

Lskip_bias_mid

    ; Conditionally apply ReLU.
    tst     w9, #4
    b.eq    Lskip_relu_mid
    fmax    v0.4s, v0.4s, v31.4s
    fmax    v1.4s, v1.4s, v31.4s
    fmax    v2.4s, v2.4s, v31.4s
    fmax    v3.4s, v3.4s, v31.4s

Lskip_relu_mid

    ; Store the result.
    stp     q0, q1, [x12, #0]
    stp     q2, q3, [x12, #32]

    ; Advance filter/output/bias pointers for the next filter set block.
    add     x4, x4, x25
    add     x5, x5, x26
    add     x6, x6, #64
    subs    x3, x3, #1
    b.ne    Lfilter_mid_loop

    ; Advance to the next output index.
    add     x1, x1, x22
    add     x2, x2, #64
    subs    x0, x0, #1
    b.ne    Loutput_mid_loop

    ; Process the right padded output region with bounds checks.
Loutput_right_begin
    ldr     x0, [sp, #240]
    cbz     x0, Lepilogue

Loutput_right_loop

    ; Initialize per-filter-set pointers and loop counter.
    mov     x3, x24
    mov     x4, x20
    mov     x5, x21
    mov     x6, x17

Lfilter_right_loop

    ; Clear accumulators.
    eor     v0.16b, v0.16b, v0.16b
    eor     v1.16b, v1.16b, v1.16b
    eor     v2.16b, v2.16b, v2.16b
    eor     v3.16b, v3.16b, v3.16b

    ; Convolution loop with bounds checks.
        CONV_LOOP_PAD lb5

    ; Compute the output pointer for this filter set and output index.
    add     x12, x5, x2

    ; Conditionally accumulate the existing output.
    ldr     w9, [sp, #256]
    tst     w9, #1
    b.eq    Lskip_accumulate_right
    ldp     q16, q17, [x12, #0]
    ldp     q18, q19, [x12, #32]
    fadd    v0.4s, v0.4s, v16.4s
    fadd    v1.4s, v1.4s, v17.4s
    fadd    v2.4s, v2.4s, v18.4s
    fadd    v3.4s, v3.4s, v19.4s

Lskip_accumulate_right

    ; Conditionally add bias.
    tst     w9, #2
    b.eq    Lskip_bias_right
    ldp     q16, q17, [x6, #0]
    ldp     q18, q19, [x6, #32]
    fadd    v0.4s, v0.4s, v16.4s
    fadd    v1.4s, v1.4s, v17.4s
    fadd    v2.4s, v2.4s, v18.4s
    fadd    v3.4s, v3.4s, v19.4s

Lskip_bias_right

    ; Conditionally apply ReLU.
    tst     w9, #4
    b.eq    Lskip_relu_right
    fmax    v0.4s, v0.4s, v31.4s
    fmax    v1.4s, v1.4s, v31.4s
    fmax    v2.4s, v2.4s, v31.4s
    fmax    v3.4s, v3.4s, v31.4s

Lskip_relu_right

    ; Store the result.
    stp     q0, q1, [x12, #0]
    stp     q2, q3, [x12, #32]

    ; Advance filter/output/bias pointers for the next filter set block.
    add     x4, x4, x25
    add     x5, x5, x26
    add     x6, x6, #64
    subs    x3, x3, #1
    b.ne    Lfilter_right_loop

    ; Advance to the next output index.
    add     x1, x1, x22
    add     x2, x2, #64
    subs    x0, x0, #1
    b.ne    Loutput_right_loop

    ; Skip the flag-free fast path section.
    b       Lepilogue

Lkernel_flags0

    ; KernelFlags == 0 fast path: no accumulation, bias, or activation.

    ; Process the left padded output region with bounds checks.
    ldr     x0, [sp, #224]
    cbz     x0, Loutput_mid_begin_flags0

    ; Pair up the padded outputs so that one filter load serves two of them.
    ; The padded columns are a small fraction of the row but are computed one
    ; output at a time, so they cost about three times what an interior column
    ; does; for Out=7x7 shapes they are two of the seven columns.
    and     x16, x0, #1                  ; Odd output left over after pairs.
    str     x16, [sp, #264]
    lsr     x0, x0, #1                   ; Pair count.
    cbz     x0, Loutput_left_single_begin_flags0

Loutput_left_pair_loop_flags0

    ; Initialize per-filter-set pointers and loop counter.
    mov     x3, x24
    mov     x4, x20
    mov     x5, x21

Lfilter_left_pair_loop_flags0

    ; Clear accumulators for both output points.
    eor     v0.16b, v0.16b, v0.16b
    eor     v1.16b, v1.16b, v1.16b
    eor     v2.16b, v2.16b, v2.16b
    eor     v3.16b, v3.16b, v3.16b
    eor     v20.16b, v20.16b, v20.16b
    eor     v21.16b, v21.16b, v21.16b
    eor     v22.16b, v22.16b, v22.16b
    eor     v23.16b, v23.16b, v23.16b

    ; Convolution loop with bounds checks computing two outputs.
        CONV_LOOP_PAD_2OUT lb13

    ; Compute the output pointers for the two output points.
    add     x12, x5, x2
    add     x11, x12, #64

    ; Store the results for the two output points.
    stp     q0, q1, [x12, #0]
    stp     q2, q3, [x12, #32]
    stp     q20, q21, [x11, #0]
    stp     q22, q23, [x11, #32]

    ; Advance filter/output pointers for the next filter set block.
    add     x4, x4, x25
    add     x5, x5, x26
    subs    x3, x3, #1
    b.ne    Lfilter_left_pair_loop_flags0

    ; Advance to the next two output indices.
    add     x1, x1, x22, lsl #1
    add     x2, x2, #128
    subs    x0, x0, #1
    b.ne    Loutput_left_pair_loop_flags0

Loutput_left_single_begin_flags0
    ldr     x0, [sp, #264]
    cbz     x0, Loutput_mid_begin_flags0

Loutput_left_loop_flags0

    ; Initialize per-filter-set pointers and loop counter.
    mov     x3, x24
    mov     x4, x20
    mov     x5, x21

Lfilter_left_loop_flags0

    ; Clear accumulators.
    eor     v0.16b, v0.16b, v0.16b
    eor     v1.16b, v1.16b, v1.16b
    eor     v2.16b, v2.16b, v2.16b
    eor     v3.16b, v3.16b, v3.16b

    ; Convolution loop with bounds checks.
        CONV_LOOP_PAD lb6

    ; Compute the output pointer for this filter set and output index.
    add     x12, x5, x2

    ; Store the result.
    stp     q0, q1, [x12, #0]
    stp     q2, q3, [x12, #32]

    ; Advance filter/output pointers for the next filter set block.
    add     x4, x4, x25
    add     x5, x5, x26
    subs    x3, x3, #1
    b.ne    Lfilter_left_loop_flags0

    ; Advance to the next output index.
    add     x1, x1, x22
    add     x2, x2, #64
    subs    x0, x0, #1
    b.ne    Loutput_left_loop_flags0

    ; Process the middle output region without bounds checks.
Loutput_mid_begin_flags0
    ldr     x0, [sp, #232]
    cbz     x0, Loutput_right_begin_flags0

    ; Process four outputs at a time to amortize the filter loads.
    and     x16, x0, #3                  ; Remainder outputs after quads.
    str     x16, [sp, #248]
    lsr     x0, x0, #2                   ; Quad output count.
    cbz     x0, Loutput_mid_remainder_begin_flags0

Loutput_mid_quad_loop_flags0

    ; Initialize per-filter-set pointers and loop counter.
    mov     x3, x24
    mov     x4, x20
    mov     x5, x21

Lfilter_mid_quad_loop_flags0

    ; Clear accumulators for four output points.
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

    ; Convolution loop without bounds checks computing four outputs.
        CONV_LOOP_MID_4OUT lb7, x6

    ; Compute the output pointers for the four output points.
    add     x12, x5, x2
    add     x11, x12, #64
    add     x10, x12, #128
    add     x9,  x12, #192

    ; Store the results for the four output points.
    stp     q0, q1, [x12, #0]
    stp     q2, q3, [x12, #32]
    stp     q4, q5, [x11, #0]
    stp     q6, q7, [x11, #32]
    stp     q8, q9, [x10, #0]
    stp     q10, q11, [x10, #32]
    stp     q12, q13, [x9, #0]
    stp     q14, q15, [x9, #32]

    ; Advance filter/output pointers for the next filter set block.
    add     x4, x4, x25
    add     x5, x5, x26
    subs    x3, x3, #1
    b.ne    Lfilter_mid_quad_loop_flags0

    ; Advance to the next four output indices.
    add     x1, x1, x22, lsl #2
    add     x2, x2, #256
    subs    x0, x0, #1
    b.ne    Loutput_mid_quad_loop_flags0

Loutput_mid_remainder_begin_flags0
    ldr     x0, [sp, #248]
    cbz     x0, Loutput_right_begin_flags0
    cmp     x0, #3
    b.ne    Loutput_mid_remainder_not3_flags0
    ; Exactly three outputs remain.
    str     xzr, [sp, #248]
    mov     x0, #1
    b       Loutput_mid_triad_loop_flags0

Loutput_mid_remainder_not3_flags0
    cmp     x0, #2
    b.ne    Loutput_mid_remainder_single_flags0
    ; Exactly two outputs remain.
    str     xzr, [sp, #248]
    mov     x0, #1
    b       Loutput_mid_pair_loop_flags0

Loutput_mid_remainder_single_flags0
    ; Exactly one output remains.
    mov     x0, #1
    b       Loutput_mid_loop_flags0

Loutput_mid_triad_loop_flags0

    ; Initialize per-filter-set pointers and loop counter.
    mov     x3, x24
    mov     x4, x20
    mov     x5, x21

Lfilter_mid_triad_loop_flags0

    ; Clear accumulators for three output points.
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

    ; Convolution loop without bounds checks computing three outputs.
        CONV_LOOP_MID_3OUT lb8

    ; Compute the output pointers for the three output points.
    add     x12, x5, x2
    add     x11, x12, #64
    add     x10, x12, #128

    ; Store the results for the three output points.
    stp     q0, q1, [x12, #0]
    stp     q2, q3, [x12, #32]
    stp     q4, q5, [x11, #0]
    stp     q6, q7, [x11, #32]
    stp     q8, q9, [x10, #0]
    stp     q10, q11, [x10, #32]

    ; Advance filter/output pointers for the next filter set block.
    add     x4, x4, x25
    add     x5, x5, x26
    subs    x3, x3, #1
    b.ne    Lfilter_mid_triad_loop_flags0

    ; Advance to the next three output indices.
    add     x1, x1, x22, lsl #1
    add     x1, x1, x22
    add     x2, x2, #192
    subs    x0, x0, #1
    b.ne    Loutput_mid_triad_loop_flags0

Loutput_mid_pair_begin_flags0
    ldr     x0, [sp, #248]
    and     x16, x0, #1
    str     x16, [sp, #248]
    lsr     x0, x0, #1
    cbz     x0, Loutput_mid_single_begin_flags0

Loutput_mid_pair_loop_flags0

    ; Initialize per-filter-set pointers and loop counter.
    mov     x3, x24
    mov     x4, x20
    mov     x5, x21

Lfilter_mid_pair_loop_flags0

    ; Clear accumulators for both output points.
    eor     v0.16b, v0.16b, v0.16b
    eor     v1.16b, v1.16b, v1.16b
    eor     v2.16b, v2.16b, v2.16b
    eor     v3.16b, v3.16b, v3.16b
    eor     v20.16b, v20.16b, v20.16b
    eor     v21.16b, v21.16b, v21.16b
    eor     v22.16b, v22.16b, v22.16b
    eor     v23.16b, v23.16b, v23.16b

    ; Convolution loop without bounds checks computing two outputs.
        CONV_LOOP_MID_2OUT lb9

    ; Compute the output pointers for the two output points.
    add     x12, x5, x2
    add     x11, x12, #64

    ; Store the results for the two output points.
    stp     q0, q1, [x12, #0]
    stp     q2, q3, [x12, #32]
    stp     q20, q21, [x11, #0]
    stp     q22, q23, [x11, #32]

    ; Advance filter/output pointers for the next filter set block.
    add     x4, x4, x25
    add     x5, x5, x26
    subs    x3, x3, #1
    b.ne    Lfilter_mid_pair_loop_flags0

    ; Advance to the next two output indices.
    add     x1, x1, x22, lsl #1
    add     x2, x2, #128
    subs    x0, x0, #1
    b.ne    Loutput_mid_pair_loop_flags0

Loutput_mid_single_begin_flags0
    ldr     x0, [sp, #248]
    cbz     x0, Loutput_right_begin_flags0

Loutput_mid_loop_flags0

    ; Initialize per-filter-set pointers and loop counter.
    mov     x3, x24
    mov     x4, x20
    mov     x5, x21

Lfilter_mid_loop_flags0

    ; Clear accumulators.
    eor     v0.16b, v0.16b, v0.16b
    eor     v1.16b, v1.16b, v1.16b
    eor     v2.16b, v2.16b, v2.16b
    eor     v3.16b, v3.16b, v3.16b

    ; Convolution loop without bounds checks.
        CONV_LOOP_MID lb10

    ; Compute the output pointer for this filter set and output index.
    add     x12, x5, x2

    ; Store the result.
    stp     q0, q1, [x12, #0]
    stp     q2, q3, [x12, #32]

    ; Advance filter/output pointers for the next filter set block.
    add     x4, x4, x25
    add     x5, x5, x26
    subs    x3, x3, #1
    b.ne    Lfilter_mid_loop_flags0

    ; Advance to the next output index.
    add     x1, x1, x22
    add     x2, x2, #64
    subs    x0, x0, #1
    b.ne    Loutput_mid_loop_flags0

    ; Process the right padded output region with bounds checks.
Loutput_right_begin_flags0
    ldr     x0, [sp, #240]
    cbz     x0, Lepilogue

Loutput_right_loop_flags0

    ; Initialize per-filter-set pointers and loop counter.
    mov     x3, x24
    mov     x4, x20
    mov     x5, x21

Lfilter_right_loop_flags0

    ; Clear accumulators.
    eor     v0.16b, v0.16b, v0.16b
    eor     v1.16b, v1.16b, v1.16b
    eor     v2.16b, v2.16b, v2.16b
    eor     v3.16b, v3.16b, v3.16b

    ; Convolution loop with bounds checks.
        CONV_LOOP_PAD lb11

    ; Compute the output pointer for this filter set and output index.
    add     x12, x5, x2

    ; Store the result.
    stp     q0, q1, [x12, #0]
    stp     q2, q3, [x12, #32]

    ; Advance filter/output pointers for the next filter set block.
    add     x4, x4, x25
    add     x5, x5, x26
    subs    x3, x3, #1
    b.ne    Lfilter_right_loop_flags0

    ; Advance to the next output index.
    add     x1, x1, x22
    add     x2, x2, #64
    subs    x0, x0, #1
    b.ne    Loutput_right_loop_flags0

    b       Lepilogue

Lepilogue

    ; Epilogue and callee-saved register restore.
    EPILOG_NOP ldp     q14, q15, [sp, #192]
    EPILOG_NOP ldp     q12, q13, [sp, #160]
    EPILOG_NOP ldp     q10, q11, [sp, #128]
    EPILOG_NOP ldp     q8, q9, [sp, #96]
    EPILOG_RESTORE_REG_PAIR x27, x28, #80
    EPILOG_RESTORE_REG_PAIR x25, x26, #64
    EPILOG_RESTORE_REG_PAIR x23, x24, #48
    EPILOG_RESTORE_REG_PAIR x21, x22, #32
    EPILOG_RESTORE_REG_PAIR x19, x20, #16
    EPILOG_RESTORE_REG_PAIR x29, x30, #272!
    EPILOG_RETURN

        NESTED_END MlasConvNchwcFloatKernelNeonAsm

        END
