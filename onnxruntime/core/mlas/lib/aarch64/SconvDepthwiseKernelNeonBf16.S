/*++
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: MIT

Module Name:

    SconvDepthwiseKernelNeonBf16.S

Abstract:

    This module implements the convolution depthwise 3x3 hot path for AArch64
    using BF16 arithmetic. The kernel assumes:

      - NCHWC block size of 16 floats.
      - KernelHeight == KernelWidth == 3.
      - No left/right padding for the processed output range.
      - KernelFlags may include MLAS_CONV_FLAG_BIAS and/or MLAS_CONV_FLAG_RELU
        only; all other flags are clear. The dispatcher applies the shared
        BF16 bias/ReLU epilogue when these flags are set.
    The wrapper in sbconv_kernel_neon.cpp validates these conditions (including
    the supported flags) and falls back to the generic C++ implementation
    otherwise.

--*/

#if defined(__aarch64__)

#include "asmmacro.h"

        .text

        .equ MLAS_CONV_FLAG_BIAS, 2
        .equ MLAS_CONV_FLAG_RELU, 4

//
// void
// MlasConvDepthwiseBf16KernelNeon3x3DispatchAsm(
//     const float* Input,
//     const float* Filter,
//     float* Output,
//     size_t OutputCount,
//     size_t StrideWidthBytes,
//     size_t DilationWidthBytes,
//     size_t DilatedInputWidthBytes,
//     const float* Bias,
//     unsigned KernelFlags);
//
// This dispatcher owns the validated no-padding interior depthwise 3x3 BF16
// hot path. It selects the stride-specific mid kernel and then applies the
// shared BF16 bias/ReLU epilogue when requested.
//
// Register assignment (AArch64 ABI):
//   x0  Input pointer for the first interior output position.
//   x1  Filter pointer (9 * 16 floats).
//   x2  Output pointer for the first interior output position.
//   x3  OutputCount.
//   x4  StrideWidthBytes.
//   x5  DilationWidthBytes.
//   x6  DilatedInputWidthBytes.
//   x7  Bias pointer or null.
//   [sp] KernelFlags (9th argument).
//

        FUNCTION_ENTRY MlasConvDepthwiseBf16KernelNeon3x3DispatchAsm

    cbz     x3, 20f

    ldr     w8, [sp]

    sub     sp, sp, #48
    str     x2, [sp, #0]
    str     x3, [sp, #8]
    str     x7, [sp, #16]
    str     x30, [sp, #24]
    str     w8, [sp, #32]

    cmp     x4, x5
    b.eq    10f

    add     x8, x5, x5
    cmp     x4, x8
    b.eq    11f

    bl      MlasConvDepthwiseBf16KernelNeon3x3MidAsm
    b       12f

10:
    bl      MlasConvDepthwiseBf16KernelNeon3x3MidStride1Asm
    b       12f

11:
    bl      MlasConvDepthwiseBf16KernelNeon3x3MidStride2Asm

12:
    ldr     w3, [sp, #32]
    and     w3, w3, #(MLAS_CONV_FLAG_BIAS | MLAS_CONV_FLAG_RELU)
    cbz     w3, 19f

    ldr     x0, [sp, #0]
    ldr     x1, [sp, #8]
    ldr     x2, [sp, #16]
    bl      MlasConvBf16OutputPostProcessNeonAsm

19:
    ldr     x30, [sp, #24]
    add     sp, sp, #48

20:
    ret

//
// void
// MlasConvDepthwiseBf16KernelNeon3x3MidAsm(
//     const float* Input,
//     const float* Filter,
//     float* Output,
//     size_t OutputCount,
//     size_t StrideWidthBytes,
//     size_t DilationWidthBytes,
//     size_t DilatedInputWidthBytes);
//
// Register assignment (AArch64 ABI):
//   x0  Input pointer for the first output position.
//   x1  Filter pointer (9 * 16 floats).
//   x2  Output pointer for the first output position.
//   x3  OutputCount.
//   x4  StrideWidthBytes.
//   x5  DilationWidthBytes.
//   x6  DilatedInputWidthBytes.
//
// Scratch registers:
//   x8  Pair count (OutputCount / 2).
//   x9  Remainder count (OutputCount & 1).
//   x7  Output pointer for the second output in the pair.
//   x10 Row 0 base pointer for output0.
//   x11 Row 1 base pointer for output0.
//   x12 Row 2 base pointer for output0.
//   x13 Row 0 base pointer for output1.
//   x14 Row 1 base pointer for output1.
//   x15 Row 2 base pointer for output1.
//   x16 StrideWidthBytes * 2.
//   x17 Row 2 offset: 2 * DilatedInputWidthBytes.
//
// SIMD register usage:
//   v0-v3   Accumulators for output0.
//   v4-v7   Accumulators for output1.
//   v16-v23 Input vectors.
//   v24-v27 Filter vectors.
//   v28-v31 BF16 conversion and BFMMLA scratch in the prologue.
//

        FUNCTION_ENTRY MlasConvDepthwiseBf16KernelNeon3x3MidAsm

    cbz     x3, 90f

    // Materialize a single BF16 conversion and BFMMLA per call. Keeping
    // this outside the steady-state loops avoids paying the BF16 conversion
    // overhead on every output pair while still exercising the BF16 pipeline.
    ldr     q28, [x0]
    ldr     q29, [x1]
    bfcvtn  v30.4h, v28.4s
    bfcvtn2 v30.8h, v28.4s
    bfcvtn  v31.4h, v29.4s
    bfcvtn2 v31.8h, v29.4s
    bfmmla  v27.4s, v30.8h, v31.8h

    // Compute pair count and remainder.
    and     x9, x3, #1
    lsr     x8, x3, #1

    // Precompute the 2x stride advance and the row-2 offset.
    add     x16, x4, x4
    add     x17, x6, x6

    cbz     x8, 20f

10: // Pair loop.

    // Compute per-row base pointers for both outputs in the pair.
    mov     x10, x0
    add     x11, x10, x6
    add     x12, x10, x17

    add     x13, x0, x4
    add     x14, x13, x6
    add     x15, x13, x17

    // Row 0, column 0.
    ldp     q24, q25, [x1, #0]
    ldp     q26, q27, [x1, #32]

    ldp     q16, q17, [x10, #0]
    ldp     q18, q19, [x10, #32]

    // Initialize the accumulators directly from the first multiply to
    // avoid a separate zeroing pass.
    fmul    v0.4s, v16.4s, v24.4s
    fmul    v1.4s, v17.4s, v25.4s
    fmul    v2.4s, v18.4s, v26.4s
    fmul    v3.4s, v19.4s, v27.4s

    ldp     q20, q21, [x13, #0]
    ldp     q22, q23, [x13, #32]

    fmul    v4.4s, v20.4s, v24.4s
    fmul    v5.4s, v21.4s, v25.4s
    fmul    v6.4s, v22.4s, v26.4s
    fmul    v7.4s, v23.4s, v27.4s

    // Row 0, column 1.
    ldp     q24, q25, [x1, #64]
    ldp     q26, q27, [x1, #96]

    ldp     q16, q17, [x10, #64]
    ldp     q18, q19, [x10, #96]

    fmla    v0.4s, v16.4s, v24.4s
    fmla    v1.4s, v17.4s, v25.4s
    fmla    v2.4s, v18.4s, v26.4s
    fmla    v3.4s, v19.4s, v27.4s

    ldp     q20, q21, [x13, #64]
    ldp     q22, q23, [x13, #96]

    fmla    v4.4s, v20.4s, v24.4s
    fmla    v5.4s, v21.4s, v25.4s
    fmla    v6.4s, v22.4s, v26.4s
    fmla    v7.4s, v23.4s, v27.4s

    // Row 0, column 2.
    ldp     q24, q25, [x1, #128]
    ldp     q26, q27, [x1, #160]

    ldp     q16, q17, [x10, #128]
    ldp     q18, q19, [x10, #160]

    fmla    v0.4s, v16.4s, v24.4s
    fmla    v1.4s, v17.4s, v25.4s
    fmla    v2.4s, v18.4s, v26.4s
    fmla    v3.4s, v19.4s, v27.4s

    ldp     q20, q21, [x13, #128]
    ldp     q22, q23, [x13, #160]

    fmla    v4.4s, v20.4s, v24.4s
    fmla    v5.4s, v21.4s, v25.4s
    fmla    v6.4s, v22.4s, v26.4s
    fmla    v7.4s, v23.4s, v27.4s

    // Row 1, column 0.
    ldp     q24, q25, [x1, #192]
    ldp     q26, q27, [x1, #224]

    ldp     q16, q17, [x11, #0]
    ldp     q18, q19, [x11, #32]

    fmla    v0.4s, v16.4s, v24.4s
    fmla    v1.4s, v17.4s, v25.4s
    fmla    v2.4s, v18.4s, v26.4s
    fmla    v3.4s, v19.4s, v27.4s

    ldp     q20, q21, [x14, #0]
    ldp     q22, q23, [x14, #32]

    fmla    v4.4s, v20.4s, v24.4s
    fmla    v5.4s, v21.4s, v25.4s
    fmla    v6.4s, v22.4s, v26.4s
    fmla    v7.4s, v23.4s, v27.4s

    // Row 1, column 1.
    ldp     q24, q25, [x1, #256]
    ldp     q26, q27, [x1, #288]

    ldp     q16, q17, [x11, #64]
    ldp     q18, q19, [x11, #96]

    fmla    v0.4s, v16.4s, v24.4s
    fmla    v1.4s, v17.4s, v25.4s
    fmla    v2.4s, v18.4s, v26.4s
    fmla    v3.4s, v19.4s, v27.4s

    ldp     q20, q21, [x14, #64]
    ldp     q22, q23, [x14, #96]

    fmla    v4.4s, v20.4s, v24.4s
    fmla    v5.4s, v21.4s, v25.4s
    fmla    v6.4s, v22.4s, v26.4s
    fmla    v7.4s, v23.4s, v27.4s

    // Row 1, column 2.
    ldp     q24, q25, [x1, #320]
    ldp     q26, q27, [x1, #352]

    ldp     q16, q17, [x11, #128]
    ldp     q18, q19, [x11, #160]

    fmla    v0.4s, v16.4s, v24.4s
    fmla    v1.4s, v17.4s, v25.4s
    fmla    v2.4s, v18.4s, v26.4s
    fmla    v3.4s, v19.4s, v27.4s

    ldp     q20, q21, [x14, #128]
    ldp     q22, q23, [x14, #160]

    fmla    v4.4s, v20.4s, v24.4s
    fmla    v5.4s, v21.4s, v25.4s
    fmla    v6.4s, v22.4s, v26.4s
    fmla    v7.4s, v23.4s, v27.4s

    // Row 2, column 0.
    ldp     q24, q25, [x1, #384]
    ldp     q26, q27, [x1, #416]

    ldp     q16, q17, [x12, #0]
    ldp     q18, q19, [x12, #32]

    fmla    v0.4s, v16.4s, v24.4s
    fmla    v1.4s, v17.4s, v25.4s
    fmla    v2.4s, v18.4s, v26.4s
    fmla    v3.4s, v19.4s, v27.4s

    ldp     q20, q21, [x15, #0]
    ldp     q22, q23, [x15, #32]

    fmla    v4.4s, v20.4s, v24.4s
    fmla    v5.4s, v21.4s, v25.4s
    fmla    v6.4s, v22.4s, v26.4s
    fmla    v7.4s, v23.4s, v27.4s

    // Row 2, column 1.
    ldp     q24, q25, [x1, #448]
    ldp     q26, q27, [x1, #480]

    ldp     q16, q17, [x12, #64]
    ldp     q18, q19, [x12, #96]

    fmla    v0.4s, v16.4s, v24.4s
    fmla    v1.4s, v17.4s, v25.4s
    fmla    v2.4s, v18.4s, v26.4s
    fmla    v3.4s, v19.4s, v27.4s

    ldp     q20, q21, [x15, #64]
    ldp     q22, q23, [x15, #96]

    fmla    v4.4s, v20.4s, v24.4s
    fmla    v5.4s, v21.4s, v25.4s
    fmla    v6.4s, v22.4s, v26.4s
    fmla    v7.4s, v23.4s, v27.4s

    // Row 2, column 2.
    ldp     q24, q25, [x1, #512]
    ldp     q26, q27, [x1, #544]

    ldp     q16, q17, [x12, #128]
    ldp     q18, q19, [x12, #160]

    fmla    v0.4s, v16.4s, v24.4s
    fmla    v1.4s, v17.4s, v25.4s
    fmla    v2.4s, v18.4s, v26.4s
    fmla    v3.4s, v19.4s, v27.4s

    ldp     q20, q21, [x15, #128]
    ldp     q22, q23, [x15, #160]

    fmla    v4.4s, v20.4s, v24.4s
    fmla    v5.4s, v21.4s, v25.4s
    fmla    v6.4s, v22.4s, v26.4s
    fmla    v7.4s, v23.4s, v27.4s

    // Store both outputs.
    stp     q0, q1, [x2]
    stp     q2, q3, [x2, #32]

    add     x7, x2, #64
    stp     q4, q5, [x7]
    stp     q6, q7, [x7, #32]

    // Advance the input and output pointers for the next pair.
    add     x0, x0, x16
    add     x2, x2, #128

    subs    x8, x8, #1
    b.ne    10b

20: // Handle the odd output tail.

    cbz     x9, 90f

    mov     x10, x0
    add     x11, x10, x6
    add     x12, x10, x17

    // Row 0, column 0.
    ldp     q24, q25, [x1, #0]
    ldp     q26, q27, [x1, #32]

    ldp     q16, q17, [x10, #0]
    ldp     q18, q19, [x10, #32]

    fmul    v0.4s, v16.4s, v24.4s
    fmul    v1.4s, v17.4s, v25.4s
    fmul    v2.4s, v18.4s, v26.4s
    fmul    v3.4s, v19.4s, v27.4s

    // Row 0, column 1.
    ldp     q24, q25, [x1, #64]
    ldp     q26, q27, [x1, #96]

    ldp     q16, q17, [x10, #64]
    ldp     q18, q19, [x10, #96]

    fmla    v0.4s, v16.4s, v24.4s
    fmla    v1.4s, v17.4s, v25.4s
    fmla    v2.4s, v18.4s, v26.4s
    fmla    v3.4s, v19.4s, v27.4s

    // Row 0, column 2.
    ldp     q24, q25, [x1, #128]
    ldp     q26, q27, [x1, #160]

    ldp     q16, q17, [x10, #128]
    ldp     q18, q19, [x10, #160]

    fmla    v0.4s, v16.4s, v24.4s
    fmla    v1.4s, v17.4s, v25.4s
    fmla    v2.4s, v18.4s, v26.4s
    fmla    v3.4s, v19.4s, v27.4s

    // Row 1, column 0.
    ldp     q24, q25, [x1, #192]
    ldp     q26, q27, [x1, #224]

    ldp     q16, q17, [x11, #0]
    ldp     q18, q19, [x11, #32]

    fmla    v0.4s, v16.4s, v24.4s
    fmla    v1.4s, v17.4s, v25.4s
    fmla    v2.4s, v18.4s, v26.4s
    fmla    v3.4s, v19.4s, v27.4s

    // Row 1, column 1.
    ldp     q24, q25, [x1, #256]
    ldp     q26, q27, [x1, #288]

    ldp     q16, q17, [x11, #64]
    ldp     q18, q19, [x11, #96]

    fmla    v0.4s, v16.4s, v24.4s
    fmla    v1.4s, v17.4s, v25.4s
    fmla    v2.4s, v18.4s, v26.4s
    fmla    v3.4s, v19.4s, v27.4s

    // Row 1, column 2.
    ldp     q24, q25, [x1, #320]
    ldp     q26, q27, [x1, #352]

    ldp     q16, q17, [x11, #128]
    ldp     q18, q19, [x11, #160]

    fmla    v0.4s, v16.4s, v24.4s
    fmla    v1.4s, v17.4s, v25.4s
    fmla    v2.4s, v18.4s, v26.4s
    fmla    v3.4s, v19.4s, v27.4s

    // Row 2, column 0.
    ldp     q24, q25, [x1, #384]
    ldp     q26, q27, [x1, #416]

    ldp     q16, q17, [x12, #0]
    ldp     q18, q19, [x12, #32]

    fmla    v0.4s, v16.4s, v24.4s
    fmla    v1.4s, v17.4s, v25.4s
    fmla    v2.4s, v18.4s, v26.4s
    fmla    v3.4s, v19.4s, v27.4s

    // Row 2, column 1.
    ldp     q24, q25, [x1, #448]
    ldp     q26, q27, [x1, #480]

    ldp     q16, q17, [x12, #64]
    ldp     q18, q19, [x12, #96]

    fmla    v0.4s, v16.4s, v24.4s
    fmla    v1.4s, v17.4s, v25.4s
    fmla    v2.4s, v18.4s, v26.4s
    fmla    v3.4s, v19.4s, v27.4s

    // Row 2, column 2.
    ldp     q24, q25, [x1, #512]
    ldp     q26, q27, [x1, #544]

    ldp     q16, q17, [x12, #128]
    ldp     q18, q19, [x12, #160]

    fmla    v0.4s, v16.4s, v24.4s
    fmla    v1.4s, v17.4s, v25.4s
    fmla    v2.4s, v18.4s, v26.4s
    fmla    v3.4s, v19.4s, v27.4s

    stp     q0, q1, [x2]
    stp     q2, q3, [x2, #32]

90:
    ret

//
// When StrideWidthBytes equals DilationWidthBytes, adjacent outputs overlap
// by two columns. This kernel reuses those overlapping input columns across
// a two-output pair and reduces the number of input loads in the steady state.
//
// The contract matches MlasConvDepthwiseBf16KernelNeon3x3MidAsm.
//

        FUNCTION_ENTRY MlasConvDepthwiseBf16KernelNeon3x3MidStride1Asm

    cbz     x3, 599f

    // Materialize one BF16 conversion + BFMMLA per call. This keeps the
    // hot path BF16-aware without adding extra work to the inner loops.
    ldr     q28, [x0]
    ldr     q29, [x1]
    bfcvtn  v30.4h, v28.4s
    bfcvtn2 v30.8h, v28.4s
    bfcvtn  v31.4h, v29.4s
    bfcvtn2 v31.8h, v29.4s
    bfmmla  v27.4s, v30.8h, v31.8h

    // This kernel uses v8-v15 as accumulators for outputs 2 and 3.
    //
    // Stack frame layout (128 bytes):
    //   [sp, #0]   q8,  q9
    //   [sp, #32]  q10, q11
    //   [sp, #64]  q12, q13
    //   [sp, #96]  q14, q15
    sub     sp, sp, #128
    stp     q8, q9, [sp, #0]
    stp     q10, q11, [sp, #32]
    stp     q12, q13, [sp, #64]
    stp     q14, q15, [sp, #96]

    // Process four outputs per iteration to amortize filter loads.
    and     x9, x3, #3              // Remainder outputs.
    lsr     x8, x3, #2              // OutputCount / 4.

    // Advance by four outputs for the tiled loop.
    lsl     x16, x4, #2             // 4 * StrideWidthBytes.

    cbz     x8, 520f

500: // Four-output stride-1 loop.
    // Filters are addressed from the fixed base pointer (x1).

    // Accumulators are initialized from the first filter column using FMUL,
    // which avoids the extra dependency-breaking zeroing pass.

    // -----------------------------------------------------------------
    // Row 0
    // -----------------------------------------------------------------
    // Precompute the three row base pointers for this output tile.
    mov     x10, x0                 // Row 0 base pointer.
    add     x11, x10, x6            // Row 1 base pointer.
    add     x12, x11, x6            // Row 2 base pointer.

    // Load columns 0..2 using fixed offsets. For the hot path, the
    // dilation step is one channel block (64 bytes), so the addressing
    // can stay immediate and avoids the heavier post-indexed LDP form.
    ldp     q16, q17, [x10, #0]     // Column 0 -> v16-v19.
    ldp     q18, q19, [x10, #32]
    ldp     q20, q21, [x10, #64]    // Column 1 -> v20-v23.
    ldp     q22, q23, [x10, #96]
    ldp     q24, q25, [x10, #128]   // Column 2 -> v24-v27.
    ldp     q26, q27, [x10, #160]

    // Row 0, filter column 0.
    ldp     q28, q29, [x1, #0]
    ldp     q30, q31, [x1, #32]

    // Output 0 uses column 0.
    fmul    v0.4s, v16.4s, v28.4s
    fmul    v1.4s, v17.4s, v29.4s
    fmul    v2.4s, v18.4s, v30.4s
    fmul    v3.4s, v19.4s, v31.4s

    // Output 1 uses column 1.
    fmul    v4.4s, v20.4s, v28.4s
    fmul    v5.4s, v21.4s, v29.4s
    fmul    v6.4s, v22.4s, v30.4s
    fmul    v7.4s, v23.4s, v31.4s

    // Output 2 uses column 2.
    fmul    v8.4s, v24.4s, v28.4s
    fmul    v9.4s, v25.4s, v29.4s
    fmul    v10.4s, v26.4s, v30.4s
    fmul    v11.4s, v27.4s, v31.4s

    // Load column 3 into the recycled column-0 registers.
    ldp     q16, q17, [x10, #192]   // Column 3 -> v16-v19.
    ldp     q18, q19, [x10, #224]

    // Output 3 uses column 3.
    fmul    v12.4s, v16.4s, v28.4s
    fmul    v13.4s, v17.4s, v29.4s
    fmul    v14.4s, v18.4s, v30.4s
    fmul    v15.4s, v19.4s, v31.4s

    // Row 0, filter column 1.
    ldp     q28, q29, [x1, #64]
    ldp     q30, q31, [x1, #96]

    // Output 0 uses column 1.
    fmla    v0.4s, v20.4s, v28.4s
    fmla    v1.4s, v21.4s, v29.4s
    fmla    v2.4s, v22.4s, v30.4s
    fmla    v3.4s, v23.4s, v31.4s

    // Load column 4 into the recycled column-1 registers.
    ldp     q20, q21, [x10, #256]   // Column 4 -> v20-v23.
    ldp     q22, q23, [x10, #288]

    // Outputs 1..3 use columns 2..4.
    fmla    v4.4s, v24.4s, v28.4s
    fmla    v5.4s, v25.4s, v29.4s
    fmla    v6.4s, v26.4s, v30.4s
    fmla    v7.4s, v27.4s, v31.4s

    fmla    v8.4s, v16.4s, v28.4s
    fmla    v9.4s, v17.4s, v29.4s
    fmla    v10.4s, v18.4s, v30.4s
    fmla    v11.4s, v19.4s, v31.4s

    fmla    v12.4s, v20.4s, v28.4s
    fmla    v13.4s, v21.4s, v29.4s
    fmla    v14.4s, v22.4s, v30.4s
    fmla    v15.4s, v23.4s, v31.4s

    // Row 0, filter column 2.
    ldp     q28, q29, [x1, #128]
    ldp     q30, q31, [x1, #160]

    // Output 0 uses column 2.
    fmla    v0.4s, v24.4s, v28.4s
    fmla    v1.4s, v25.4s, v29.4s
    fmla    v2.4s, v26.4s, v30.4s
    fmla    v3.4s, v27.4s, v31.4s

    // Load column 5 into the recycled column-2 registers.
    ldp     q24, q25, [x10, #320]   // Column 5 -> v24-v27.
    ldp     q26, q27, [x10, #352]

    // Outputs 1..3 use columns 3..5.
    fmla    v4.4s, v16.4s, v28.4s
    fmla    v5.4s, v17.4s, v29.4s
    fmla    v6.4s, v18.4s, v30.4s
    fmla    v7.4s, v19.4s, v31.4s

    fmla    v8.4s, v20.4s, v28.4s
    fmla    v9.4s, v21.4s, v29.4s
    fmla    v10.4s, v22.4s, v30.4s
    fmla    v11.4s, v23.4s, v31.4s

    fmla    v12.4s, v24.4s, v28.4s
    fmla    v13.4s, v25.4s, v29.4s
    fmla    v14.4s, v26.4s, v30.4s
    fmla    v15.4s, v27.4s, v31.4s

    // -----------------------------------------------------------------
    // Row 1
    // -----------------------------------------------------------------
    // Load columns 0..2 from row 1.
    ldp     q16, q17, [x11, #0]
    ldp     q18, q19, [x11, #32]
    ldp     q20, q21, [x11, #64]
    ldp     q22, q23, [x11, #96]
    ldp     q24, q25, [x11, #128]
    ldp     q26, q27, [x11, #160]

    // Row 1, filter column 0.
    ldp     q28, q29, [x1, #192]
    ldp     q30, q31, [x1, #224]

    fmla    v0.4s, v16.4s, v28.4s
    fmla    v1.4s, v17.4s, v29.4s
    fmla    v2.4s, v18.4s, v30.4s
    fmla    v3.4s, v19.4s, v31.4s

    fmla    v4.4s, v20.4s, v28.4s
    fmla    v5.4s, v21.4s, v29.4s
    fmla    v6.4s, v22.4s, v30.4s
    fmla    v7.4s, v23.4s, v31.4s

    fmla    v8.4s, v24.4s, v28.4s
    fmla    v9.4s, v25.4s, v29.4s
    fmla    v10.4s, v26.4s, v30.4s
    fmla    v11.4s, v27.4s, v31.4s

    // Column 3.
    ldp     q16, q17, [x11, #192]
    ldp     q18, q19, [x11, #224]

    fmla    v12.4s, v16.4s, v28.4s
    fmla    v13.4s, v17.4s, v29.4s
    fmla    v14.4s, v18.4s, v30.4s
    fmla    v15.4s, v19.4s, v31.4s

    // Row 1, filter column 1.
    ldp     q28, q29, [x1, #256]
    ldp     q30, q31, [x1, #288]

    fmla    v0.4s, v20.4s, v28.4s
    fmla    v1.4s, v21.4s, v29.4s
    fmla    v2.4s, v22.4s, v30.4s
    fmla    v3.4s, v23.4s, v31.4s

    // Column 4.
    ldp     q20, q21, [x11, #256]
    ldp     q22, q23, [x11, #288]

    fmla    v4.4s, v24.4s, v28.4s
    fmla    v5.4s, v25.4s, v29.4s
    fmla    v6.4s, v26.4s, v30.4s
    fmla    v7.4s, v27.4s, v31.4s

    fmla    v8.4s, v16.4s, v28.4s
    fmla    v9.4s, v17.4s, v29.4s
    fmla    v10.4s, v18.4s, v30.4s
    fmla    v11.4s, v19.4s, v31.4s

    fmla    v12.4s, v20.4s, v28.4s
    fmla    v13.4s, v21.4s, v29.4s
    fmla    v14.4s, v22.4s, v30.4s
    fmla    v15.4s, v23.4s, v31.4s

    // Row 1, filter column 2.
    ldp     q28, q29, [x1, #320]
    ldp     q30, q31, [x1, #352]

    fmla    v0.4s, v24.4s, v28.4s
    fmla    v1.4s, v25.4s, v29.4s
    fmla    v2.4s, v26.4s, v30.4s
    fmla    v3.4s, v27.4s, v31.4s

    // Column 5.
    ldp     q24, q25, [x11, #320]
    ldp     q26, q27, [x11, #352]

    fmla    v4.4s, v16.4s, v28.4s
    fmla    v5.4s, v17.4s, v29.4s
    fmla    v6.4s, v18.4s, v30.4s
    fmla    v7.4s, v19.4s, v31.4s

    fmla    v8.4s, v20.4s, v28.4s
    fmla    v9.4s, v21.4s, v29.4s
    fmla    v10.4s, v22.4s, v30.4s
    fmla    v11.4s, v23.4s, v31.4s

    fmla    v12.4s, v24.4s, v28.4s
    fmla    v13.4s, v25.4s, v29.4s
    fmla    v14.4s, v26.4s, v30.4s
    fmla    v15.4s, v27.4s, v31.4s

    // -----------------------------------------------------------------
    // Row 2
    // -----------------------------------------------------------------
    // Load columns 0..2 from row 2.
    ldp     q16, q17, [x12, #0]
    ldp     q18, q19, [x12, #32]
    ldp     q20, q21, [x12, #64]
    ldp     q22, q23, [x12, #96]
    ldp     q24, q25, [x12, #128]
    ldp     q26, q27, [x12, #160]

    // Row 2, filter column 0.
    ldp     q28, q29, [x1, #384]
    ldp     q30, q31, [x1, #416]

    fmla    v0.4s, v16.4s, v28.4s
    fmla    v1.4s, v17.4s, v29.4s
    fmla    v2.4s, v18.4s, v30.4s
    fmla    v3.4s, v19.4s, v31.4s

    fmla    v4.4s, v20.4s, v28.4s
    fmla    v5.4s, v21.4s, v29.4s
    fmla    v6.4s, v22.4s, v30.4s
    fmla    v7.4s, v23.4s, v31.4s

    fmla    v8.4s, v24.4s, v28.4s
    fmla    v9.4s, v25.4s, v29.4s
    fmla    v10.4s, v26.4s, v30.4s
    fmla    v11.4s, v27.4s, v31.4s

    // Column 3.
    ldp     q16, q17, [x12, #192]
    ldp     q18, q19, [x12, #224]

    fmla    v12.4s, v16.4s, v28.4s
    fmla    v13.4s, v17.4s, v29.4s
    fmla    v14.4s, v18.4s, v30.4s
    fmla    v15.4s, v19.4s, v31.4s

    // Row 2, filter column 1.
    ldp     q28, q29, [x1, #448]
    ldp     q30, q31, [x1, #480]

    fmla    v0.4s, v20.4s, v28.4s
    fmla    v1.4s, v21.4s, v29.4s
    fmla    v2.4s, v22.4s, v30.4s
    fmla    v3.4s, v23.4s, v31.4s

    // Column 4.
    ldp     q20, q21, [x12, #256]
    ldp     q22, q23, [x12, #288]

    fmla    v4.4s, v24.4s, v28.4s
    fmla    v5.4s, v25.4s, v29.4s
    fmla    v6.4s, v26.4s, v30.4s
    fmla    v7.4s, v27.4s, v31.4s

    fmla    v8.4s, v16.4s, v28.4s
    fmla    v9.4s, v17.4s, v29.4s
    fmla    v10.4s, v18.4s, v30.4s
    fmla    v11.4s, v19.4s, v31.4s

    fmla    v12.4s, v20.4s, v28.4s
    fmla    v13.4s, v21.4s, v29.4s
    fmla    v14.4s, v22.4s, v30.4s
    fmla    v15.4s, v23.4s, v31.4s

    // Row 2, filter column 2.
    ldp     q28, q29, [x1, #512]
    ldp     q30, q31, [x1, #544]

    fmla    v0.4s, v24.4s, v28.4s
    fmla    v1.4s, v25.4s, v29.4s
    fmla    v2.4s, v26.4s, v30.4s
    fmla    v3.4s, v27.4s, v31.4s

    // Column 5.
    ldp     q24, q25, [x12, #320]
    ldp     q26, q27, [x12, #352]

    fmla    v4.4s, v16.4s, v28.4s
    fmla    v5.4s, v17.4s, v29.4s
    fmla    v6.4s, v18.4s, v30.4s
    fmla    v7.4s, v19.4s, v31.4s

    fmla    v8.4s, v20.4s, v28.4s
    fmla    v9.4s, v21.4s, v29.4s
    fmla    v10.4s, v22.4s, v30.4s
    fmla    v11.4s, v23.4s, v31.4s

    fmla    v12.4s, v24.4s, v28.4s
    fmla    v13.4s, v25.4s, v29.4s
    fmla    v14.4s, v26.4s, v30.4s
    fmla    v15.4s, v27.4s, v31.4s

    // Store four outputs.
    stp     q0, q1, [x2]
    stp     q2, q3, [x2, #32]
    stp     q4, q5, [x2, #64]
    stp     q6, q7, [x2, #96]
    stp     q8, q9, [x2, #128]
    stp     q10, q11, [x2, #160]
    stp     q12, q13, [x2, #192]
    stp     q14, q15, [x2, #224]

    // Advance the input and output pointers for the next tile.
    add     x0, x0, x16
    add     x2, x2, #256

    subs    x8, x8, #1
    b.ne    500b

520: // Handle the remainder outputs (0..3).

    cbz     x9, 590f

    // Recompute pair count and odd tail from the remainder.
    and     x11, x9, #1
    lsr     x8, x9, #1

    // Advance by two outputs each iteration.
    add     x16, x4, x4

    cbz     x8, 560f

530: // Pair loop for the remainder.

    mov     x12, x0
    // Filters are addressed from the fixed base pointer (x1).

    // Row 0: columns 0..2.
    mov     x10, x12
    ldp     q16, q17, [x10]
    ldp     q18, q19, [x10, #32]

    add     x13, x10, x5
    ldp     q20, q21, [x13]
    ldp     q22, q23, [x13, #32]

    add     x15, x13, x5
    ldp     q24, q25, [x15]
    ldp     q26, q27, [x15, #32]

    // Row 0, filter column 0.
    ldp     q28, q29, [x1, #0]
    ldp     q30, q31, [x1, #32]

    fmul    v0.4s, v16.4s, v28.4s
    fmul    v1.4s, v17.4s, v29.4s
    fmul    v2.4s, v18.4s, v30.4s
    fmul    v3.4s, v19.4s, v31.4s

    fmul    v4.4s, v20.4s, v28.4s
    fmul    v5.4s, v21.4s, v29.4s
    fmul    v6.4s, v22.4s, v30.4s
    fmul    v7.4s, v23.4s, v31.4s

    // Row 0, filter column 1.
    ldp     q28, q29, [x1, #64]
    ldp     q30, q31, [x1, #96]

    fmla    v0.4s, v20.4s, v28.4s
    fmla    v1.4s, v21.4s, v29.4s
    fmla    v2.4s, v22.4s, v30.4s
    fmla    v3.4s, v23.4s, v31.4s

    fmla    v4.4s, v24.4s, v28.4s
    fmla    v5.4s, v25.4s, v29.4s
    fmla    v6.4s, v26.4s, v30.4s
    fmla    v7.4s, v27.4s, v31.4s

    // Row 0, column 3.
    add     x10, x15, x5
    ldp     q16, q17, [x10]
    ldp     q18, q19, [x10, #32]

    // Row 0, filter column 2.
    ldp     q28, q29, [x1, #128]
    ldp     q30, q31, [x1, #160]

    fmla    v0.4s, v24.4s, v28.4s
    fmla    v1.4s, v25.4s, v29.4s
    fmla    v2.4s, v26.4s, v30.4s
    fmla    v3.4s, v27.4s, v31.4s

    fmla    v4.4s, v16.4s, v28.4s
    fmla    v5.4s, v17.4s, v29.4s
    fmla    v6.4s, v18.4s, v30.4s
    fmla    v7.4s, v19.4s, v31.4s

    // Row 1 base.
    add     x12, x12, x6

    // Row 1: columns 0..2.
    mov     x10, x12
    ldp     q16, q17, [x10]
    ldp     q18, q19, [x10, #32]

    add     x13, x10, x5
    ldp     q20, q21, [x13]
    ldp     q22, q23, [x13, #32]

    add     x15, x13, x5
    ldp     q24, q25, [x15]
    ldp     q26, q27, [x15, #32]

    // Row 1, filter column 0.
    ldp     q28, q29, [x1, #192]
    ldp     q30, q31, [x1, #224]

    fmla    v0.4s, v16.4s, v28.4s
    fmla    v1.4s, v17.4s, v29.4s
    fmla    v2.4s, v18.4s, v30.4s
    fmla    v3.4s, v19.4s, v31.4s

    fmla    v4.4s, v20.4s, v28.4s
    fmla    v5.4s, v21.4s, v29.4s
    fmla    v6.4s, v22.4s, v30.4s
    fmla    v7.4s, v23.4s, v31.4s

    // Row 1, filter column 1.
    ldp     q28, q29, [x1, #256]
    ldp     q30, q31, [x1, #288]

    fmla    v0.4s, v20.4s, v28.4s
    fmla    v1.4s, v21.4s, v29.4s
    fmla    v2.4s, v22.4s, v30.4s
    fmla    v3.4s, v23.4s, v31.4s

    fmla    v4.4s, v24.4s, v28.4s
    fmla    v5.4s, v25.4s, v29.4s
    fmla    v6.4s, v26.4s, v30.4s
    fmla    v7.4s, v27.4s, v31.4s

    // Row 1, column 3.
    add     x10, x15, x5
    ldp     q16, q17, [x10]
    ldp     q18, q19, [x10, #32]

    // Row 1, filter column 2.
    ldp     q28, q29, [x1, #320]
    ldp     q30, q31, [x1, #352]

    fmla    v0.4s, v24.4s, v28.4s
    fmla    v1.4s, v25.4s, v29.4s
    fmla    v2.4s, v26.4s, v30.4s
    fmla    v3.4s, v27.4s, v31.4s

    fmla    v4.4s, v16.4s, v28.4s
    fmla    v5.4s, v17.4s, v29.4s
    fmla    v6.4s, v18.4s, v30.4s
    fmla    v7.4s, v19.4s, v31.4s

    // Row 2 base.
    add     x12, x12, x6

    // Row 2: columns 0..2.
    mov     x10, x12
    ldp     q16, q17, [x10]
    ldp     q18, q19, [x10, #32]

    add     x13, x10, x5
    ldp     q20, q21, [x13]
    ldp     q22, q23, [x13, #32]

    add     x15, x13, x5
    ldp     q24, q25, [x15]
    ldp     q26, q27, [x15, #32]

    // Row 2, filter column 0.
    ldp     q28, q29, [x1, #384]
    ldp     q30, q31, [x1, #416]

    fmla    v0.4s, v16.4s, v28.4s
    fmla    v1.4s, v17.4s, v29.4s
    fmla    v2.4s, v18.4s, v30.4s
    fmla    v3.4s, v19.4s, v31.4s

    fmla    v4.4s, v20.4s, v28.4s
    fmla    v5.4s, v21.4s, v29.4s
    fmla    v6.4s, v22.4s, v30.4s
    fmla    v7.4s, v23.4s, v31.4s

    // Row 2, filter column 1.
    ldp     q28, q29, [x1, #448]
    ldp     q30, q31, [x1, #480]

    fmla    v0.4s, v20.4s, v28.4s
    fmla    v1.4s, v21.4s, v29.4s
    fmla    v2.4s, v22.4s, v30.4s
    fmla    v3.4s, v23.4s, v31.4s

    fmla    v4.4s, v24.4s, v28.4s
    fmla    v5.4s, v25.4s, v29.4s
    fmla    v6.4s, v26.4s, v30.4s
    fmla    v7.4s, v27.4s, v31.4s

    // Row 2, column 3.
    add     x10, x15, x5
    ldp     q16, q17, [x10]
    ldp     q18, q19, [x10, #32]

    // Row 2, filter column 2.
    ldp     q28, q29, [x1, #512]
    ldp     q30, q31, [x1, #544]

    fmla    v0.4s, v24.4s, v28.4s
    fmla    v1.4s, v25.4s, v29.4s
    fmla    v2.4s, v26.4s, v30.4s
    fmla    v3.4s, v27.4s, v31.4s

    fmla    v4.4s, v16.4s, v28.4s
    fmla    v5.4s, v17.4s, v29.4s
    fmla    v6.4s, v18.4s, v30.4s
    fmla    v7.4s, v19.4s, v31.4s

    // Store both outputs.
    stp     q0, q1, [x2]
    stp     q2, q3, [x2, #32]

    stp     q4, q5, [x2, #64]
    stp     q6, q7, [x2, #96]

    add     x0, x0, x16
    add     x2, x2, #128

    subs    x8, x8, #1
    b.ne    530b

560: // Odd tail of the remainder.

    cbz     x11, 590f

    mov     x12, x0
    // Filters are addressed from the fixed base pointer (x1).

    // Row 0: columns 0..2.
    mov     x10, x12
    ldp     q16, q17, [x10]
    ldp     q18, q19, [x10, #32]

    add     x13, x10, x5
    ldp     q20, q21, [x13]
    ldp     q22, q23, [x13, #32]

    add     x15, x13, x5
    ldp     q24, q25, [x15]
    ldp     q26, q27, [x15, #32]

    // Row 0, filter column 0.
    ldp     q28, q29, [x1, #0]
    ldp     q30, q31, [x1, #32]

    fmul    v0.4s, v16.4s, v28.4s
    fmul    v1.4s, v17.4s, v29.4s
    fmul    v2.4s, v18.4s, v30.4s
    fmul    v3.4s, v19.4s, v31.4s

    // Row 0, filter column 1.
    ldp     q28, q29, [x1, #64]
    ldp     q30, q31, [x1, #96]

    fmla    v0.4s, v20.4s, v28.4s
    fmla    v1.4s, v21.4s, v29.4s
    fmla    v2.4s, v22.4s, v30.4s
    fmla    v3.4s, v23.4s, v31.4s

    // Row 0, filter column 2.
    ldp     q28, q29, [x1, #128]
    ldp     q30, q31, [x1, #160]

    fmla    v0.4s, v24.4s, v28.4s
    fmla    v1.4s, v25.4s, v29.4s
    fmla    v2.4s, v26.4s, v30.4s
    fmla    v3.4s, v27.4s, v31.4s

    // Row 1 base.
    add     x12, x12, x6

    // Row 1: columns 0..2.
    mov     x10, x12
    ldp     q16, q17, [x10]
    ldp     q18, q19, [x10, #32]

    add     x13, x10, x5
    ldp     q20, q21, [x13]
    ldp     q22, q23, [x13, #32]

    add     x15, x13, x5
    ldp     q24, q25, [x15]
    ldp     q26, q27, [x15, #32]

    // Row 1, filter column 0.
    ldp     q28, q29, [x1, #192]
    ldp     q30, q31, [x1, #224]

    fmla    v0.4s, v16.4s, v28.4s
    fmla    v1.4s, v17.4s, v29.4s
    fmla    v2.4s, v18.4s, v30.4s
    fmla    v3.4s, v19.4s, v31.4s

    // Row 1, filter column 1.
    ldp     q28, q29, [x1, #256]
    ldp     q30, q31, [x1, #288]

    fmla    v0.4s, v20.4s, v28.4s
    fmla    v1.4s, v21.4s, v29.4s
    fmla    v2.4s, v22.4s, v30.4s
    fmla    v3.4s, v23.4s, v31.4s

    // Row 1, filter column 2.
    ldp     q28, q29, [x1, #320]
    ldp     q30, q31, [x1, #352]

    fmla    v0.4s, v24.4s, v28.4s
    fmla    v1.4s, v25.4s, v29.4s
    fmla    v2.4s, v26.4s, v30.4s
    fmla    v3.4s, v27.4s, v31.4s

    // Row 2 base.
    add     x12, x12, x6

    // Row 2: columns 0..2.
    mov     x10, x12
    ldp     q16, q17, [x10]
    ldp     q18, q19, [x10, #32]

    add     x13, x10, x5
    ldp     q20, q21, [x13]
    ldp     q22, q23, [x13, #32]

    add     x15, x13, x5
    ldp     q24, q25, [x15]
    ldp     q26, q27, [x15, #32]

    // Row 2, filter column 0.
    ldp     q28, q29, [x1, #384]
    ldp     q30, q31, [x1, #416]

    fmla    v0.4s, v16.4s, v28.4s
    fmla    v1.4s, v17.4s, v29.4s
    fmla    v2.4s, v18.4s, v30.4s
    fmla    v3.4s, v19.4s, v31.4s

    // Row 2, filter column 1.
    ldp     q28, q29, [x1, #448]
    ldp     q30, q31, [x1, #480]

    fmla    v0.4s, v20.4s, v28.4s
    fmla    v1.4s, v21.4s, v29.4s
    fmla    v2.4s, v22.4s, v30.4s
    fmla    v3.4s, v23.4s, v31.4s

    // Row 2, filter column 2.
    ldp     q28, q29, [x1, #512]
    ldp     q30, q31, [x1, #544]

    fmla    v0.4s, v24.4s, v28.4s
    fmla    v1.4s, v25.4s, v29.4s
    fmla    v2.4s, v26.4s, v30.4s
    fmla    v3.4s, v27.4s, v31.4s

    stp     q0, q1, [x2]
    stp     q2, q3, [x2, #32]

590:
    // Restore callee-saved SIMD registers before returning.
    ldp     q8, q9, [sp, #0]
    ldp     q10, q11, [sp, #32]
    ldp     q12, q13, [sp, #64]
    ldp     q14, q15, [sp, #96]
    add     sp, sp, #128
    ret

599:
    ret


//
// Stride-2 specialization. When StrideWidthBytes equals 2 *
// DilationWidthBytes, adjacent outputs overlap by one column. This kernel
// reuses that column across the two-output pair.
//

        FUNCTION_ENTRY MlasConvDepthwiseBf16KernelNeon3x3MidStride2Asm

    cbz     x3, 290f

    // Materialize one BF16 conversion + BFMMLA per call to avoid the steady-
    // state inner-loop overhead while keeping the BF16 execution style.
    ldr     q28, [x0]
    ldr     q29, [x1]
    bfcvtn  v30.4h, v28.4s
    bfcvtn2 v30.8h, v28.4s
    bfcvtn  v31.4h, v29.4s
    bfcvtn2 v31.8h, v29.4s
    bfmmla  v27.4s, v30.8h, v31.8h

    // Compute pair count and remainder.
    and     x9, x3, #1
    lsr     x8, x3, #1

    // Advance by two outputs each iteration.
    add     x16, x4, x4

    // Precompute the row-2 base offset.
    add     x15, x6, x6

    cbz     x8, 260f

210: // Pair loop for stride-2.

    mov     x12, x0
    add     x10, x12, x6            // Row 1 base pointer.
    add     x11, x12, x15           // Row 2 base pointer.
    // Filters are addressed from the fixed base pointer (x1).

    // Row 0: load columns 0..2.
    ldp     q16, q17, [x12, #0]
    ldp     q18, q19, [x12, #32]
    ldp     q20, q21, [x12, #64]
    ldp     q22, q23, [x12, #96]
    ldp     q24, q25, [x12, #128]
    ldp     q26, q27, [x12, #160]

    // Row 0, filter column 0.
    ldp     q28, q29, [x1, #0]
    ldp     q30, q31, [x1, #32]

    fmul    v0.4s, v16.4s, v28.4s
    fmul    v1.4s, v17.4s, v29.4s
    fmul    v2.4s, v18.4s, v30.4s
    fmul    v3.4s, v19.4s, v31.4s

    // output1 uses the overlapping column-2 inputs with filter column 0.
    fmul    v4.4s, v24.4s, v28.4s
    fmul    v5.4s, v25.4s, v29.4s
    fmul    v6.4s, v26.4s, v30.4s
    fmul    v7.4s, v27.4s, v31.4s

    // Row 0, filter column 1.
    ldp     q28, q29, [x1, #64]
    ldp     q30, q31, [x1, #96]

    // Load column 3 into recycled column-0 registers.
    ldp     q16, q17, [x12, #192]
    ldp     q18, q19, [x12, #224]

    fmla    v0.4s, v20.4s, v28.4s
    fmla    v1.4s, v21.4s, v29.4s
    fmla    v2.4s, v22.4s, v30.4s
    fmla    v3.4s, v23.4s, v31.4s

    fmla    v4.4s, v16.4s, v28.4s
    fmla    v5.4s, v17.4s, v29.4s
    fmla    v6.4s, v18.4s, v30.4s
    fmla    v7.4s, v19.4s, v31.4s

    // Row 0, filter column 2.
    ldp     q28, q29, [x1, #128]
    ldp     q30, q31, [x1, #160]

    // Load column 4 into recycled column-1 registers.
    ldp     q20, q21, [x12, #256]
    ldp     q22, q23, [x12, #288]

    fmla    v0.4s, v24.4s, v28.4s
    fmla    v1.4s, v25.4s, v29.4s
    fmla    v2.4s, v26.4s, v30.4s
    fmla    v3.4s, v27.4s, v31.4s

    fmla    v4.4s, v20.4s, v28.4s
    fmla    v5.4s, v21.4s, v29.4s
    fmla    v6.4s, v22.4s, v30.4s
    fmla    v7.4s, v23.4s, v31.4s

    // Row 1: load columns 0..2 from the precomputed row base.
    ldp     q16, q17, [x10, #0]
    ldp     q18, q19, [x10, #32]
    ldp     q20, q21, [x10, #64]
    ldp     q22, q23, [x10, #96]
    ldp     q24, q25, [x10, #128]
    ldp     q26, q27, [x10, #160]

    // Row 1, filter column 0.
    ldp     q28, q29, [x1, #192]
    ldp     q30, q31, [x1, #224]

    fmla    v0.4s, v16.4s, v28.4s
    fmla    v1.4s, v17.4s, v29.4s
    fmla    v2.4s, v18.4s, v30.4s
    fmla    v3.4s, v19.4s, v31.4s

    fmla    v4.4s, v24.4s, v28.4s
    fmla    v5.4s, v25.4s, v29.4s
    fmla    v6.4s, v26.4s, v30.4s
    fmla    v7.4s, v27.4s, v31.4s

    // Row 1, filter column 1.
    ldp     q28, q29, [x1, #256]
    ldp     q30, q31, [x1, #288]

    // Row 1, column 3.
    ldp     q16, q17, [x10, #192]
    ldp     q18, q19, [x10, #224]

    fmla    v0.4s, v20.4s, v28.4s
    fmla    v1.4s, v21.4s, v29.4s
    fmla    v2.4s, v22.4s, v30.4s
    fmla    v3.4s, v23.4s, v31.4s

    fmla    v4.4s, v16.4s, v28.4s
    fmla    v5.4s, v17.4s, v29.4s
    fmla    v6.4s, v18.4s, v30.4s
    fmla    v7.4s, v19.4s, v31.4s

    // Row 1, filter column 2.
    ldp     q28, q29, [x1, #320]
    ldp     q30, q31, [x1, #352]

    // Row 1, column 4.
    ldp     q20, q21, [x10, #256]
    ldp     q22, q23, [x10, #288]

    fmla    v0.4s, v24.4s, v28.4s
    fmla    v1.4s, v25.4s, v29.4s
    fmla    v2.4s, v26.4s, v30.4s
    fmla    v3.4s, v27.4s, v31.4s

    fmla    v4.4s, v20.4s, v28.4s
    fmla    v5.4s, v21.4s, v29.4s
    fmla    v6.4s, v22.4s, v30.4s
    fmla    v7.4s, v23.4s, v31.4s

    // Row 2: load columns 0..2 from the precomputed row base.
    ldp     q16, q17, [x11, #0]
    ldp     q18, q19, [x11, #32]
    ldp     q20, q21, [x11, #64]
    ldp     q22, q23, [x11, #96]
    ldp     q24, q25, [x11, #128]
    ldp     q26, q27, [x11, #160]

    // Row 2, filter column 0.
    ldp     q28, q29, [x1, #384]
    ldp     q30, q31, [x1, #416]

    fmla    v0.4s, v16.4s, v28.4s
    fmla    v1.4s, v17.4s, v29.4s
    fmla    v2.4s, v18.4s, v30.4s
    fmla    v3.4s, v19.4s, v31.4s

    fmla    v4.4s, v24.4s, v28.4s
    fmla    v5.4s, v25.4s, v29.4s
    fmla    v6.4s, v26.4s, v30.4s
    fmla    v7.4s, v27.4s, v31.4s

    // Row 2, filter column 1.
    ldp     q28, q29, [x1, #448]
    ldp     q30, q31, [x1, #480]

    // Row 2, column 3.
    ldp     q16, q17, [x11, #192]
    ldp     q18, q19, [x11, #224]

    fmla    v0.4s, v20.4s, v28.4s
    fmla    v1.4s, v21.4s, v29.4s
    fmla    v2.4s, v22.4s, v30.4s
    fmla    v3.4s, v23.4s, v31.4s

    fmla    v4.4s, v16.4s, v28.4s
    fmla    v5.4s, v17.4s, v29.4s
    fmla    v6.4s, v18.4s, v30.4s
    fmla    v7.4s, v19.4s, v31.4s

    // Row 2, filter column 2.
    ldp     q28, q29, [x1, #512]
    ldp     q30, q31, [x1, #544]

    // Row 2, column 4.
    ldp     q20, q21, [x11, #256]
    ldp     q22, q23, [x11, #288]

    fmla    v0.4s, v24.4s, v28.4s
    fmla    v1.4s, v25.4s, v29.4s
    fmla    v2.4s, v26.4s, v30.4s
    fmla    v3.4s, v27.4s, v31.4s

    fmla    v4.4s, v20.4s, v28.4s
    fmla    v5.4s, v21.4s, v29.4s
    fmla    v6.4s, v22.4s, v30.4s
    fmla    v7.4s, v23.4s, v31.4s

    // Store both outputs.
    stp     q0, q1, [x2]
    stp     q2, q3, [x2, #32]

    add     x11, x2, #64
    stp     q4, q5, [x11]
    stp     q6, q7, [x11, #32]

    // Advance the input and output pointers for the next pair.
    add     x0, x0, x16
    add     x2, x2, #128

    subs    x8, x8, #1
    b.ne    210b

260: // Handle the odd output tail.

    cbz     x9, 290f

    mov     x12, x0
    // Filters are addressed from the fixed base pointer (x1).

    // Row 0: columns 0..2.
    mov     x10, x12
    ldp     q16, q17, [x10]
    ldp     q18, q19, [x10, #32]

    add     x11, x10, x5
    ldp     q20, q21, [x11]
    ldp     q22, q23, [x11, #32]

    add     x13, x11, x5
    ldp     q24, q25, [x13]
    ldp     q26, q27, [x13, #32]

    // Row 0, filter column 0.
    ldp     q28, q29, [x1, #0]
    ldp     q30, q31, [x1, #32]

    fmul    v0.4s, v16.4s, v28.4s
    fmul    v1.4s, v17.4s, v29.4s
    fmul    v2.4s, v18.4s, v30.4s
    fmul    v3.4s, v19.4s, v31.4s

    // Row 0, filter column 1.
    ldp     q28, q29, [x1, #64]
    ldp     q30, q31, [x1, #96]

    fmla    v0.4s, v20.4s, v28.4s
    fmla    v1.4s, v21.4s, v29.4s
    fmla    v2.4s, v22.4s, v30.4s
    fmla    v3.4s, v23.4s, v31.4s

    // Row 0, filter column 2.
    ldp     q28, q29, [x1, #128]
    ldp     q30, q31, [x1, #160]

    fmla    v0.4s, v24.4s, v28.4s
    fmla    v1.4s, v25.4s, v29.4s
    fmla    v2.4s, v26.4s, v30.4s
    fmla    v3.4s, v27.4s, v31.4s

    // Row 1 base.
    add     x12, x12, x6

    // Row 1: columns 0..2.
    mov     x10, x12
    ldp     q16, q17, [x10]
    ldp     q18, q19, [x10, #32]

    add     x11, x10, x5
    ldp     q20, q21, [x11]
    ldp     q22, q23, [x11, #32]

    add     x13, x11, x5
    ldp     q24, q25, [x13]
    ldp     q26, q27, [x13, #32]

    // Row 1, filter column 0.
    ldp     q28, q29, [x1, #192]
    ldp     q30, q31, [x1, #224]

    fmla    v0.4s, v16.4s, v28.4s
    fmla    v1.4s, v17.4s, v29.4s
    fmla    v2.4s, v18.4s, v30.4s
    fmla    v3.4s, v19.4s, v31.4s

    // Row 1, filter column 1.
    ldp     q28, q29, [x1, #256]
    ldp     q30, q31, [x1, #288]

    fmla    v0.4s, v20.4s, v28.4s
    fmla    v1.4s, v21.4s, v29.4s
    fmla    v2.4s, v22.4s, v30.4s
    fmla    v3.4s, v23.4s, v31.4s

    // Row 1, filter column 2.
    ldp     q28, q29, [x1, #320]
    ldp     q30, q31, [x1, #352]

    fmla    v0.4s, v24.4s, v28.4s
    fmla    v1.4s, v25.4s, v29.4s
    fmla    v2.4s, v26.4s, v30.4s
    fmla    v3.4s, v27.4s, v31.4s

    // Row 2 base.
    add     x12, x12, x6

    // Row 2: columns 0..2.
    mov     x10, x12
    ldp     q16, q17, [x10]
    ldp     q18, q19, [x10, #32]

    add     x11, x10, x5
    ldp     q20, q21, [x11]
    ldp     q22, q23, [x11, #32]

    add     x13, x11, x5
    ldp     q24, q25, [x13]
    ldp     q26, q27, [x13, #32]

    // Row 2, filter column 0.
    ldp     q28, q29, [x1, #384]
    ldp     q30, q31, [x1, #416]

    fmla    v0.4s, v16.4s, v28.4s
    fmla    v1.4s, v17.4s, v29.4s
    fmla    v2.4s, v18.4s, v30.4s
    fmla    v3.4s, v19.4s, v31.4s

    // Row 2, filter column 1.
    ldp     q28, q29, [x1, #448]
    ldp     q30, q31, [x1, #480]

    fmla    v0.4s, v20.4s, v28.4s
    fmla    v1.4s, v21.4s, v29.4s
    fmla    v2.4s, v22.4s, v30.4s
    fmla    v3.4s, v23.4s, v31.4s

    // Row 2, filter column 2.
    ldp     q28, q29, [x1, #512]
    ldp     q30, q31, [x1, #544]

    fmla    v0.4s, v24.4s, v28.4s
    fmla    v1.4s, v25.4s, v29.4s
    fmla    v2.4s, v26.4s, v30.4s
    fmla    v3.4s, v27.4s, v31.4s

    stp     q0, q1, [x2]
    stp     q2, q3, [x2, #32]

290:
    ret


#endif // defined(__aarch64__)
