/*++
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: MIT

Module Name:

    SconvKernelNeonBf16.S

Abstract:

    This module implements the direct NCHW convolution hot path for AArch64
    using BF16 BFMMLA instructions with FP32 accumulation.

--*/

#include "asmmacro.h"

        .text
        .arch_extension bf16

// MLAS_NCHW_BF16_MMLA_PARAMS offsets (validated in C++ with static_assert).
        .equ MLAS_NCHW_INPUT, 0
        .equ MLAS_NCHW_PACKED_FILTER, 8
        .equ MLAS_NCHW_OUTPUT, 16
        .equ MLAS_NCHW_BIAS, 24
        .equ MLAS_NCHW_STRIDE_W, 32
        .equ MLAS_NCHW_DILATED_W, 40
        .equ MLAS_NCHW_OUTPUT_COUNT, 48
        .equ MLAS_NCHW_FILTER_COUNT, 56
        .equ MLAS_NCHW_OUTPUT_STRIDE, 64
        .equ MLAS_NCHW_KERNEL_FLAGS, 72

// Packed filter layout constants.
        .equ MLAS_BF16_GROUP_STRIDE_BYTES, 128
        .equ MLAS_BF16_GROUP2_OFFSET_BYTES, 256
        .equ MLAS_FILTER_CHANNEL_PAIR_BYTES, 16
        .equ MLAS_OUTPUT_CHANNEL_PAIR_BYTES, 8
        .equ MLAS_FILTER_CHANNEL_PAIR2_BYTES, 32
        .equ MLAS_OUTPUT_CHANNEL_PAIR2_BYTES, 16
        .equ MLAS_BF16_GROUP_STRIDE_PAIR1_BYTES, 144
        .equ MLAS_BF16_GROUP2_PAIR1_BYTES, 272

// Output layout constants.
        .equ MLAS_OUTPUT_BLOCK_BYTES, 64
        .equ MLAS_OUTPUT_PAIR_BYTES, 128

// Packed filter stride for one filter set (192 bf16 values).
        .equ MLAS_BF16_FILTER_STRIDE_BYTES, 384
        .equ MLAS_BF16_FILTER_STRIDE_ADJUST_BYTES, 256

// Convolution kernel flags.
        .equ MLAS_CONV_FLAG_ACCUMULATE, 1
        .equ MLAS_CONV_FLAG_BIAS, 2
        .equ MLAS_CONV_FLAG_RELU, 4

//
// void MLASCALL
// MlasConvNchwBf16KernelNeonAsm(const MLAS_NCHW_BF16_MMLA_PARAMS* Params)
//
// Registers:
//   x1  - input pointer (current output pair)
//   x2  - packed filter base pointer
//   x3  - output base pointer for filter block 0 (current output pair)
//   x4  - bias base pointer for filter block 0
//   x5  - stride width in bytes
//   x6  - dilated input width in bytes (row stride)
//   x7  - output pair count
//   x9  - output stride in bytes between filter blocks
//   x10 - filter count
//   x15 - two-output stride in bytes
//

FUNCTION_ENTRY MlasConvNchwBf16KernelNeonAsm
        ldr     x1, [x0, #MLAS_NCHW_INPUT]
        ldr     x2, [x0, #MLAS_NCHW_PACKED_FILTER]
        ldr     x3, [x0, #MLAS_NCHW_OUTPUT]
        ldr     x4, [x0, #MLAS_NCHW_BIAS]
        ldr     x5, [x0, #MLAS_NCHW_STRIDE_W]
        ldr     x6, [x0, #MLAS_NCHW_DILATED_W]
        ldr     x7, [x0, #MLAS_NCHW_OUTPUT_COUNT]
        ldr     x10, [x0, #MLAS_NCHW_FILTER_COUNT]
        ldr     x9, [x0, #MLAS_NCHW_OUTPUT_STRIDE]
        ldr     w8, [x0, #MLAS_NCHW_KERNEL_FLAGS]

        cbz     x7, 199f
        cbz     x10, 199f

        lsl     x5, x5, #2                      // stride width bytes
        lsl     x6, x6, #2                      // row stride bytes
        lsl     x9, x9, #2                      // output stride bytes
        add     x15, x5, x5                     // two-output stride bytes

        lsr     x7, x7, #1                      // number of output pairs
        cbz     x7, 199f

        movi    v31.16b, #0                     // zero vector

        // If bias flag is set but the bias pointer is null, ignore the bias flag.
        tst     w8, #MLAS_CONV_FLAG_BIAS
        b.eq    10f
        cbnz    x4, 10f
        bic     w8, w8, #MLAS_CONV_FLAG_BIAS
10:
        // Dispatch to the most common flag combinations first.
        tst     w8, #(MLAS_CONV_FLAG_ACCUMULATE | MLAS_CONV_FLAG_BIAS | MLAS_CONV_FLAG_RELU)
        b.eq    100f

        tst     w8, #MLAS_CONV_FLAG_ACCUMULATE
        b.ne    180f

        tst     w8, #MLAS_CONV_FLAG_BIAS
        b.ne    20f

        tst     w8, #MLAS_CONV_FLAG_RELU
        b.ne    160f
        b       100f
20:
        tst     w8, #MLAS_CONV_FLAG_RELU
        b.ne    140f
        b       120f

//
// Fast path: no accumulation, no bias, no activation.
//
100:
101:
        // Compute the three BF16 input groups for two output positions.
        mov     x11, x1                         // row0, output0
        add     x12, x1, x6                     // row1, output0
        add     x13, x12, x6                    // row2, output0
        add     x14, x1, x5                     // row0, output1
        add     x0, x12, x5                     // row1, output1

        ld1     {v0.4s}, [x11]
        ld1     {v1.4s}, [x14]
        ld1     {v2.4s}, [x12]
        ld1     {v3.4s}, [x0]
        ld1     {v4.4s}, [x13]

        add     x11, x13, x5                    // row2, output1
        ld1     {v5.4s}, [x11]

        // Group 0: row0[0..2], row1[0]
        ins     v0.s[3], v2.s[0]
        ins     v1.s[3], v3.s[0]
        bfcvtn  v24.4h, v0.4s
        bfcvtn2 v24.8h, v1.4s

        // Group 1: row1[1..2], row2[0..1]
        ext     v6.16b, v4.16b, v4.16b, #8
        ext     v7.16b, v5.16b, v5.16b, #8
        ins     v6.s[0], v2.s[1]
        ins     v6.s[1], v2.s[2]
        ins     v7.s[0], v3.s[1]
        ins     v7.s[1], v3.s[2]
        bfcvtn  v25.4h, v6.4s
        bfcvtn2 v25.8h, v7.4s

        // Group 2: row2[2], zeros
        mov     v27.16b, v31.16b
        mov     v28.16b, v31.16b
        ins     v27.s[0], v4.s[2]
        ins     v28.s[0], v5.s[2]
        bfcvtn  v26.4h, v27.4s
        bfcvtn2 v26.8h, v28.4s

        // Initialize per-filter pointers.
        mov     x12, x2                         // packed filter base
        mov     x13, x3                         // output base (filter 0)
        mov     x14, x4                         // bias base (unused here)
        mov     x11, x10                        // filter count

102:
        add     x16, x13, #MLAS_OUTPUT_BLOCK_BYTES
        mov     x17, x13                        // output0/output1 pointers
        mov     x0, #4                          // 8 channel pairs, unrolled by 2

103:
        movi    v16.4s, #0
        movi    v17.4s, #0

        // Load two channel pairs of packed filter weights (three groups each).
        ldp     q1, q4, [x12, #MLAS_BF16_GROUP_STRIDE_BYTES]
        ldp     q2, q5, [x12, #MLAS_BF16_GROUP2_OFFSET_BYTES]
        ldp     q0, q3, [x12], #MLAS_FILTER_CHANNEL_PAIR2_BYTES

        // Accumulate two independent channel pairs to hide BFMMLA latency.
        bfmmla  v16.4s, v0.8h, v24.8h
        bfmmla  v16.4s, v1.8h, v25.8h
        bfmmla  v17.4s, v3.8h, v24.8h
        bfmmla  v16.4s, v2.8h, v26.8h
        bfmmla  v17.4s, v4.8h, v25.8h
        bfmmla  v17.4s, v5.8h, v26.8h

        // De-interleave channels for each output position and store 4 floats.
        uzp1    v18.4s, v16.4s, v16.4s
        uzp1    v19.4s, v17.4s, v17.4s
        uzp2    v20.4s, v16.4s, v16.4s
        uzp2    v21.4s, v17.4s, v17.4s

        st1     {v18.2s, v19.2s}, [x17], #MLAS_OUTPUT_CHANNEL_PAIR2_BYTES
        st1     {v20.2s, v21.2s}, [x16], #MLAS_OUTPUT_CHANNEL_PAIR2_BYTES

        subs    x0, x0, #1
        b.ne    103b

        add     x13, x13, x9
        add     x12, x12, #MLAS_BF16_FILTER_STRIDE_ADJUST_BYTES
        subs    x11, x11, #1
        b.ne    102b

        add     x3, x3, #MLAS_OUTPUT_PAIR_BYTES
        add     x1, x1, x15

        subs    x7, x7, #1
        b.ne    101b

        b       199f

//
// Bias only: no accumulation, no activation.
//
120:
121:
        // Compute the three BF16 input groups for two output positions.
        mov     x11, x1                         // row0, output0
        add     x12, x1, x6                     // row1, output0
        add     x13, x12, x6                    // row2, output0
        add     x14, x1, x5                     // row0, output1
        add     x0, x12, x5                     // row1, output1

        ld1     {v0.4s}, [x11]
        ld1     {v1.4s}, [x14]
        ld1     {v2.4s}, [x12]
        ld1     {v3.4s}, [x0]
        ld1     {v4.4s}, [x13]

        add     x11, x13, x5                    // row2, output1
        ld1     {v5.4s}, [x11]

        // Group 0: row0[0..2], row1[0]
        ins     v0.s[3], v2.s[0]
        ins     v1.s[3], v3.s[0]
        bfcvtn  v24.4h, v0.4s
        bfcvtn2 v24.8h, v1.4s

        // Group 1: row1[1..2], row2[0..1]
        ext     v6.16b, v4.16b, v4.16b, #8
        ext     v7.16b, v5.16b, v5.16b, #8
        ins     v6.s[0], v2.s[1]
        ins     v6.s[1], v2.s[2]
        ins     v7.s[0], v3.s[1]
        ins     v7.s[1], v3.s[2]
        bfcvtn  v25.4h, v6.4s
        bfcvtn2 v25.8h, v7.4s

        // Group 2: row2[2], zeros
        mov     v27.16b, v31.16b
        mov     v28.16b, v31.16b
        ins     v27.s[0], v4.s[2]
        ins     v28.s[0], v5.s[2]
        bfcvtn  v26.4h, v27.4s
        bfcvtn2 v26.8h, v28.4s

        // Initialize per-filter pointers.
        mov     x12, x2                         // packed filter base
        mov     x13, x3                         // output base (filter 0)
        mov     x14, x4                         // bias base
        mov     x11, x10                        // filter count

122:
        add     x16, x13, #MLAS_OUTPUT_BLOCK_BYTES
        mov     x17, x13
        mov     x0, #4

123:
        // Bias is shared across the two output positions.
        ldr     q27, [x14], #MLAS_OUTPUT_CHANNEL_PAIR2_BYTES
        zip1    v16.4s, v27.4s, v27.4s
        zip2    v17.4s, v27.4s, v27.4s

        // Load two channel pairs of packed filter weights (three groups each).
        ldp     q1, q4, [x12, #MLAS_BF16_GROUP_STRIDE_BYTES]
        ldp     q2, q5, [x12, #MLAS_BF16_GROUP2_OFFSET_BYTES]
        ldp     q0, q3, [x12], #MLAS_FILTER_CHANNEL_PAIR2_BYTES

        bfmmla  v16.4s, v0.8h, v24.8h
        bfmmla  v16.4s, v1.8h, v25.8h
        bfmmla  v17.4s, v3.8h, v24.8h
        bfmmla  v16.4s, v2.8h, v26.8h
        bfmmla  v17.4s, v4.8h, v25.8h
        bfmmla  v17.4s, v5.8h, v26.8h

        uzp1    v18.4s, v16.4s, v16.4s
        uzp1    v19.4s, v17.4s, v17.4s
        uzp2    v20.4s, v16.4s, v16.4s
        uzp2    v21.4s, v17.4s, v17.4s

        st1     {v18.2s, v19.2s}, [x17], #MLAS_OUTPUT_CHANNEL_PAIR2_BYTES
        st1     {v20.2s, v21.2s}, [x16], #MLAS_OUTPUT_CHANNEL_PAIR2_BYTES

        subs    x0, x0, #1
        b.ne    123b

        add     x13, x13, x9
        add     x12, x12, #MLAS_BF16_FILTER_STRIDE_ADJUST_BYTES
        subs    x11, x11, #1
        b.ne    122b

        add     x3, x3, #MLAS_OUTPUT_PAIR_BYTES
        add     x1, x1, x15

        subs    x7, x7, #1
        b.ne    121b

        b       199f

//
// Bias + ReLU: no accumulation.
//
140:
141:
        // Compute the three BF16 input groups for two output positions.
        mov     x11, x1                         // row0, output0
        add     x12, x1, x6                     // row1, output0
        add     x13, x12, x6                    // row2, output0
        add     x14, x1, x5                     // row0, output1
        add     x0, x12, x5                     // row1, output1

        ld1     {v0.4s}, [x11]
        ld1     {v1.4s}, [x14]
        ld1     {v2.4s}, [x12]
        ld1     {v3.4s}, [x0]
        ld1     {v4.4s}, [x13]

        add     x11, x13, x5                    // row2, output1
        ld1     {v5.4s}, [x11]

        // Group 0: row0[0..2], row1[0]
        ins     v0.s[3], v2.s[0]
        ins     v1.s[3], v3.s[0]
        bfcvtn  v24.4h, v0.4s
        bfcvtn2 v24.8h, v1.4s

        // Group 1: row1[1..2], row2[0..1]
        ext     v6.16b, v4.16b, v4.16b, #8
        ext     v7.16b, v5.16b, v5.16b, #8
        ins     v6.s[0], v2.s[1]
        ins     v6.s[1], v2.s[2]
        ins     v7.s[0], v3.s[1]
        ins     v7.s[1], v3.s[2]
        bfcvtn  v25.4h, v6.4s
        bfcvtn2 v25.8h, v7.4s

        // Group 2: row2[2], zeros
        mov     v27.16b, v31.16b
        mov     v28.16b, v31.16b
        ins     v27.s[0], v4.s[2]
        ins     v28.s[0], v5.s[2]
        bfcvtn  v26.4h, v27.4s
        bfcvtn2 v26.8h, v28.4s

        // Initialize per-filter pointers.
        mov     x12, x2                         // packed filter base
        mov     x13, x3                         // output base (filter 0)
        mov     x14, x4                         // bias base
        mov     x11, x10                        // filter count

142:
        add     x16, x13, #MLAS_OUTPUT_BLOCK_BYTES
        mov     x17, x13
        mov     x0, #4

143:
        ldr     q27, [x14], #MLAS_OUTPUT_CHANNEL_PAIR2_BYTES
        zip1    v16.4s, v27.4s, v27.4s
        zip2    v17.4s, v27.4s, v27.4s

        // Load two channel pairs of packed filter weights (three groups each).
        ldp     q1, q4, [x12, #MLAS_BF16_GROUP_STRIDE_BYTES]
        ldp     q2, q5, [x12, #MLAS_BF16_GROUP2_OFFSET_BYTES]
        ldp     q0, q3, [x12], #MLAS_FILTER_CHANNEL_PAIR2_BYTES

        bfmmla  v16.4s, v0.8h, v24.8h
        bfmmla  v16.4s, v1.8h, v25.8h
        bfmmla  v17.4s, v3.8h, v24.8h
        bfmmla  v16.4s, v2.8h, v26.8h
        bfmmla  v17.4s, v4.8h, v25.8h
        bfmmla  v17.4s, v5.8h, v26.8h

        fmax    v16.4s, v16.4s, v31.4s
        fmax    v17.4s, v17.4s, v31.4s

        uzp1    v18.4s, v16.4s, v16.4s
        uzp1    v19.4s, v17.4s, v17.4s
        uzp2    v20.4s, v16.4s, v16.4s
        uzp2    v21.4s, v17.4s, v17.4s

        st1     {v18.2s, v19.2s}, [x17], #MLAS_OUTPUT_CHANNEL_PAIR2_BYTES
        st1     {v20.2s, v21.2s}, [x16], #MLAS_OUTPUT_CHANNEL_PAIR2_BYTES

        subs    x0, x0, #1
        b.ne    143b

        add     x13, x13, x9
        add     x12, x12, #MLAS_BF16_FILTER_STRIDE_ADJUST_BYTES
        subs    x11, x11, #1
        b.ne    142b

        add     x3, x3, #MLAS_OUTPUT_PAIR_BYTES
        add     x1, x1, x15

        subs    x7, x7, #1
        b.ne    141b

        b       199f

//
// ReLU only: no accumulation, no bias.
//
160:
161:
        // Compute the three BF16 input groups for two output positions.
        mov     x11, x1                         // row0, output0
        add     x12, x1, x6                     // row1, output0
        add     x13, x12, x6                    // row2, output0
        add     x14, x1, x5                     // row0, output1
        add     x0, x12, x5                     // row1, output1

        ld1     {v0.4s}, [x11]
        ld1     {v1.4s}, [x14]
        ld1     {v2.4s}, [x12]
        ld1     {v3.4s}, [x0]
        ld1     {v4.4s}, [x13]

        add     x11, x13, x5                    // row2, output1
        ld1     {v5.4s}, [x11]

        // Group 0: row0[0..2], row1[0]
        ins     v0.s[3], v2.s[0]
        ins     v1.s[3], v3.s[0]
        bfcvtn  v24.4h, v0.4s
        bfcvtn2 v24.8h, v1.4s

        // Group 1: row1[1..2], row2[0..1]
        ext     v6.16b, v4.16b, v4.16b, #8
        ext     v7.16b, v5.16b, v5.16b, #8
        ins     v6.s[0], v2.s[1]
        ins     v6.s[1], v2.s[2]
        ins     v7.s[0], v3.s[1]
        ins     v7.s[1], v3.s[2]
        bfcvtn  v25.4h, v6.4s
        bfcvtn2 v25.8h, v7.4s

        // Group 2: row2[2], zeros
        mov     v27.16b, v31.16b
        mov     v28.16b, v31.16b
        ins     v27.s[0], v4.s[2]
        ins     v28.s[0], v5.s[2]
        bfcvtn  v26.4h, v27.4s
        bfcvtn2 v26.8h, v28.4s

        // Initialize per-filter pointers.
        mov     x12, x2                         // packed filter base
        mov     x13, x3                         // output base (filter 0)
        mov     x14, x4                         // bias base (unused here)
        mov     x11, x10                        // filter count

162:
        add     x16, x13, #MLAS_OUTPUT_BLOCK_BYTES
        mov     x17, x13
        mov     x0, #4

163:
        movi    v16.4s, #0
        movi    v17.4s, #0

        // Load two channel pairs of packed filter weights (three groups each).
        ldp     q1, q4, [x12, #MLAS_BF16_GROUP_STRIDE_BYTES]
        ldp     q2, q5, [x12, #MLAS_BF16_GROUP2_OFFSET_BYTES]
        ldp     q0, q3, [x12], #MLAS_FILTER_CHANNEL_PAIR2_BYTES

        bfmmla  v16.4s, v0.8h, v24.8h
        bfmmla  v16.4s, v1.8h, v25.8h
        bfmmla  v17.4s, v3.8h, v24.8h
        bfmmla  v16.4s, v2.8h, v26.8h
        bfmmla  v17.4s, v4.8h, v25.8h
        bfmmla  v17.4s, v5.8h, v26.8h

        fmax    v16.4s, v16.4s, v31.4s
        fmax    v17.4s, v17.4s, v31.4s

        uzp1    v18.4s, v16.4s, v16.4s
        uzp1    v19.4s, v17.4s, v17.4s
        uzp2    v20.4s, v16.4s, v16.4s
        uzp2    v21.4s, v17.4s, v17.4s

        st1     {v18.2s, v19.2s}, [x17], #MLAS_OUTPUT_CHANNEL_PAIR2_BYTES
        st1     {v20.2s, v21.2s}, [x16], #MLAS_OUTPUT_CHANNEL_PAIR2_BYTES

        subs    x0, x0, #1
        b.ne    163b

        add     x13, x13, x9
        add     x12, x12, #MLAS_BF16_FILTER_STRIDE_ADJUST_BYTES
        subs    x11, x11, #1
        b.ne    162b

        add     x3, x3, #MLAS_OUTPUT_PAIR_BYTES
        add     x1, x1, x15

        subs    x7, x7, #1
        b.ne    161b

        b       199f

//
// General accumulation path: accumulation and optional bias/ReLU.
//
180:
181:
        // Compute the three BF16 input groups for two output positions.
        mov     x11, x1                         // row0, output0
        add     x12, x1, x6                     // row1, output0
        add     x13, x12, x6                    // row2, output0
        add     x14, x1, x5                     // row0, output1
        add     x0, x12, x5                     // row1, output1

        ld1     {v0.4s}, [x11]
        ld1     {v1.4s}, [x14]
        ld1     {v2.4s}, [x12]
        ld1     {v3.4s}, [x0]
        ld1     {v4.4s}, [x13]

        add     x11, x13, x5                    // row2, output1
        ld1     {v5.4s}, [x11]

        // Group 0: row0[0..2], row1[0]
        ins     v0.s[3], v2.s[0]
        ins     v1.s[3], v3.s[0]
        bfcvtn  v24.4h, v0.4s
        bfcvtn2 v24.8h, v1.4s

        // Group 1: row1[1..2], row2[0..1]
        ext     v6.16b, v4.16b, v4.16b, #8
        ext     v7.16b, v5.16b, v5.16b, #8
        ins     v6.s[0], v2.s[1]
        ins     v6.s[1], v2.s[2]
        ins     v7.s[0], v3.s[1]
        ins     v7.s[1], v3.s[2]
        bfcvtn  v25.4h, v6.4s
        bfcvtn2 v25.8h, v7.4s

        // Group 2: row2[2], zeros
        mov     v27.16b, v31.16b
        mov     v28.16b, v31.16b
        ins     v27.s[0], v4.s[2]
        ins     v28.s[0], v5.s[2]
        bfcvtn  v26.4h, v27.4s
        bfcvtn2 v26.8h, v28.4s

        // Initialize per-filter pointers.
        mov     x12, x2                         // packed filter base
        mov     x13, x3                         // output base (filter 0)
        mov     x14, x4                         // bias base
        mov     x11, x10                        // filter count

182:
        add     x16, x13, #MLAS_OUTPUT_BLOCK_BYTES
        mov     x17, x13
        mov     x0, #4

183:
        // Load existing output and interleave for BFMMLA accumulation.
        ld1     {v18.4s}, [x17]
        ld1     {v19.4s}, [x16]
        zip1    v16.4s, v18.4s, v19.4s
        zip2    v17.4s, v18.4s, v19.4s

        // Bias is shared across the two output positions.
        tbz     w8, #1, 184f
        ldr     q27, [x14], #MLAS_OUTPUT_CHANNEL_PAIR2_BYTES
        zip1    v28.4s, v27.4s, v27.4s
        zip2    v29.4s, v27.4s, v27.4s
        fadd    v16.4s, v16.4s, v28.4s
        fadd    v17.4s, v17.4s, v29.4s
184:

        // Load two channel pairs of packed filter weights (three groups each).
        ldp     q1, q4, [x12, #MLAS_BF16_GROUP_STRIDE_BYTES]
        ldp     q2, q5, [x12, #MLAS_BF16_GROUP2_OFFSET_BYTES]
        ldp     q0, q3, [x12], #MLAS_FILTER_CHANNEL_PAIR2_BYTES

        bfmmla  v16.4s, v0.8h, v24.8h
        bfmmla  v16.4s, v1.8h, v25.8h
        bfmmla  v17.4s, v3.8h, v24.8h
        bfmmla  v16.4s, v2.8h, v26.8h
        bfmmla  v17.4s, v4.8h, v25.8h
        bfmmla  v17.4s, v5.8h, v26.8h

        tbz     w8, #2, 185f
        fmax    v16.4s, v16.4s, v31.4s
        fmax    v17.4s, v17.4s, v31.4s
185:

        uzp1    v18.4s, v16.4s, v16.4s
        uzp1    v19.4s, v17.4s, v17.4s
        uzp2    v20.4s, v16.4s, v16.4s
        uzp2    v21.4s, v17.4s, v17.4s

        st1     {v18.2s, v19.2s}, [x17], #MLAS_OUTPUT_CHANNEL_PAIR2_BYTES
        st1     {v20.2s, v21.2s}, [x16], #MLAS_OUTPUT_CHANNEL_PAIR2_BYTES

        subs    x0, x0, #1
        b.ne    183b

        add     x13, x13, x9
        add     x12, x12, #MLAS_BF16_FILTER_STRIDE_ADJUST_BYTES
        subs    x11, x11, #1
        b.ne    182b

        add     x3, x3, #MLAS_OUTPUT_PAIR_BYTES
        add     x1, x1, x15

        subs    x7, x7, #1
        b.ne    181b

199:
        ret

//
// void MLASCALL
// MlasConvNchwBf16PackFilterNeonAsm(
//     const float* Filter,            // x0
//     size_t FilterStrideElements,    // x1
//     size_t FilterCount,             // x2
//     uint16_t* PackedFilter)         // x3
//
// Pack the direct 3x3 FP32 filter blocks into the BF16 layout consumed by the
// direct BFMMLA kernel above.
//

FUNCTION_ENTRY MlasConvNchwBf16PackFilterNeonAsm
        cbz     x2, 399f

        lsl     x1, x1, #2                      // filter stride bytes
        movi    v31.16b, #0

300:    // FilterCount loop
        mov     x4, x0                          // current filter block + channel offset
        mov     x5, x3                          // packed group 0
        add     x6, x3, #MLAS_BF16_GROUP_STRIDE_BYTES
        add     x7, x3, #MLAS_BF16_GROUP2_OFFSET_BYTES
        mov     x8, #8                          // channel pairs

301:    // Channel-pair loop
        ldr     d0, [x4]
        ldr     d1, [x4, #64]
        ldr     d2, [x4, #128]
        ldr     d3, [x4, #192]
        zip1    v16.4s, v0.4s, v1.4s
        zip1    v17.4s, v2.4s, v3.4s
        bfcvtn  v18.4h, v16.4s
        bfcvtn  v19.4h, v17.4s
        zip1    v20.4s, v18.4s, v19.4s
        str     q20, [x5], #16

        ldr     d0, [x4, #256]
        ldr     d1, [x4, #320]
        ldr     d2, [x4, #384]
        ldr     d3, [x4, #448]
        zip1    v16.4s, v0.4s, v1.4s
        zip1    v17.4s, v2.4s, v3.4s
        bfcvtn  v18.4h, v16.4s
        bfcvtn  v19.4h, v17.4s
        zip1    v20.4s, v18.4s, v19.4s
        str     q20, [x6], #16

        ldr     d0, [x4, #512]
        mov     v16.16b, v31.16b
        mov     v17.16b, v31.16b
        ins     v16.s[0], v0.s[0]
        ins     v17.s[0], v0.s[1]
        bfcvtn  v20.4h, v16.4s
        bfcvtn2 v20.8h, v17.4s
        str     q20, [x7], #16

        add     x4, x4, #8
        subs    x8, x8, #1
        b.ne    301b

        add     x0, x0, x1
        add     x3, x3, #MLAS_BF16_FILTER_STRIDE_BYTES
        subs    x2, x2, #1
        b.ne    300b

399:
        ret

//
// void MLASCALL
// MlasConvBf16OutputPostProcessNeonAsm(
//     float* Output,              // x0
//     size_t OutputCount,         // x1
//     const float* Bias,          // x2
//     unsigned KernelFlags)       // w3
//
// Apply the shared BF16 epilogue across a span of 16-float output rows.
//

FUNCTION_ENTRY MlasConvBf16OutputPostProcessNeonAsm
        cbz     x1, 499f

        tst     w3, #MLAS_CONV_FLAG_BIAS
        b.eq    410f
        cbnz    x2, 410f
        bic     w3, w3, #MLAS_CONV_FLAG_BIAS
410:
        tst     w3, #(MLAS_CONV_FLAG_BIAS | MLAS_CONV_FLAG_RELU)
        b.eq    499f

        movi    v31.16b, #0

        tst     w3, #MLAS_CONV_FLAG_BIAS
        b.eq    440f

        ldp     q24, q25, [x2]
        ldp     q26, q27, [x2, #32]

        tst     w3, #MLAS_CONV_FLAG_RELU
        b.ne    460f

420:    // Bias only
        ldp     q0, q1, [x0]
        ldp     q2, q3, [x0, #32]
        fadd    v0.4s, v0.4s, v24.4s
        fadd    v1.4s, v1.4s, v25.4s
        fadd    v2.4s, v2.4s, v26.4s
        fadd    v3.4s, v3.4s, v27.4s
        stp     q0, q1, [x0]
        stp     q2, q3, [x0, #32]
        add     x0, x0, #MLAS_OUTPUT_BLOCK_BYTES
        subs    x1, x1, #1
        b.ne    420b
        b       499f

440:    // ReLU only
        ldp     q0, q1, [x0]
        ldp     q2, q3, [x0, #32]
        fmax    v0.4s, v0.4s, v31.4s
        fmax    v1.4s, v1.4s, v31.4s
        fmax    v2.4s, v2.4s, v31.4s
        fmax    v3.4s, v3.4s, v31.4s
        stp     q0, q1, [x0]
        stp     q2, q3, [x0, #32]
        add     x0, x0, #MLAS_OUTPUT_BLOCK_BYTES
        subs    x1, x1, #1
        b.ne    440b
        b       499f

460:    // Bias + ReLU
        ldp     q0, q1, [x0]
        ldp     q2, q3, [x0, #32]
        fadd    v0.4s, v0.4s, v24.4s
        fadd    v1.4s, v1.4s, v25.4s
        fadd    v2.4s, v2.4s, v26.4s
        fadd    v3.4s, v3.4s, v27.4s
        fmax    v0.4s, v0.4s, v31.4s
        fmax    v1.4s, v1.4s, v31.4s
        fmax    v2.4s, v2.4s, v31.4s
        fmax    v3.4s, v3.4s, v31.4s
        stp     q0, q1, [x0]
        stp     q2, q3, [x0, #32]
        add     x0, x0, #MLAS_OUTPUT_BLOCK_BYTES
        subs    x1, x1, #1
        b.ne    460b

499:
        ret
