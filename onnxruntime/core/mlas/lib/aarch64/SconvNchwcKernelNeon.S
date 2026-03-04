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

// -----------------------------------------------------------------------------
// void MlasConvNchwcFloatKernelNeonAsm(...)
// Implements the convolution micro-kernel for the NCHWc input format.
// -----------------------------------------------------------------------------

        FUNCTION_ENTRY MlasConvNchwcFloatKernelNeonAsm

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

        // TODO: implement kernel body

        // epilogue -------------------------------------------------------------
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
