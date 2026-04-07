/*++
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: MIT


Module Name:

    SconvPointwiseKernelNeonBf16.S

Abstract:

    This module implements the pointwise (1x1) NCHWc convolution on AArch64
    using BF16 matrix multiply accumulate instructions.

    The kernel consumes FP32 input and filter operands, but casts them to BF16
    before the multiply while accumulating in FP32 using BFMMLA.

--*/

#include "asmmacro.h"

#if defined(__aarch64__)

        .text
        .arch_extension bf16

//
// void MlasConvPointwiseBf16PackFilterNeonAsm(
//     const float* Filter,       // x0
//     size_t InputChannels,      // x1
//     uint16_t* PackedFilter)    // x2
//
// Pack one 16x16 FP32 filter tile per input-channel block into the BFMMLA-
// friendly BF16 layout consumed by the pointwise kernels below.
//

        FUNCTION_ENTRY MlasConvPointwiseBf16PackFilterNeonAsm

    cbz     x1, 19f

10: // InputChannels loop
    mov     x3, x0              // current 16x16 FP32 filter tile
    mov     x4, x2              // current packed BF16 tile
    mov     x5, #4              // four k-groups per tile

11: // K-group loop
    ldp     q24, q25, [x3, #0]
    ldp     q26, q27, [x3, #32]
    bfcvtn  v0.4h, v24.4s
    bfcvtn2 v0.8h, v25.4s
    bfcvtn  v1.4h, v26.4s
    bfcvtn2 v1.8h, v27.4s

    ldp     q24, q25, [x3, #64]
    ldp     q26, q27, [x3, #96]
    bfcvtn  v2.4h, v24.4s
    bfcvtn2 v2.8h, v25.4s
    bfcvtn  v3.4h, v26.4s
    bfcvtn2 v3.8h, v27.4s

    ldp     q24, q25, [x3, #128]
    ldp     q26, q27, [x3, #160]
    bfcvtn  v4.4h, v24.4s
    bfcvtn2 v4.8h, v25.4s
    bfcvtn  v5.4h, v26.4s
    bfcvtn2 v5.8h, v27.4s

    ldp     q24, q25, [x3, #192]
    ldp     q26, q27, [x3, #224]
    bfcvtn  v6.4h, v24.4s
    bfcvtn2 v6.8h, v25.4s
    bfcvtn  v7.4h, v26.4s
    bfcvtn2 v7.8h, v27.4s

    zip1    v16.8h, v0.8h, v2.8h
    zip2    v17.8h, v0.8h, v2.8h
    zip1    v18.8h, v4.8h, v6.8h
    zip2    v19.8h, v4.8h, v6.8h
    zip1    v20.8h, v1.8h, v3.8h
    zip2    v21.8h, v1.8h, v3.8h
    zip1    v22.8h, v5.8h, v7.8h
    zip2    v23.8h, v5.8h, v7.8h

    zip1    v24.4s, v16.4s, v18.4s
    zip2    v25.4s, v16.4s, v18.4s
    zip1    v26.4s, v17.4s, v19.4s
    zip2    v27.4s, v17.4s, v19.4s
    zip1    v28.4s, v20.4s, v22.4s
    zip2    v29.4s, v20.4s, v22.4s
    zip1    v30.4s, v21.4s, v23.4s
    zip2    v31.4s, v21.4s, v23.4s

    stp     q24, q25, [x4], #32
    stp     q26, q27, [x4], #32
    stp     q28, q29, [x4], #32
    stp     q30, q31, [x4], #32

    add     x3, x3, #256
    subs    x5, x5, #1
    b.ne    11b

    add     x0, x0, #1024
    add     x2, x2, #512
    subs    x1, x1, #1
    b.ne    10b

19:
    ret

//
// void MlasConvPointwiseBf16PackedInputKernelNeon4xAsm(
//     const uint16_t* PackedInput,    // x0
//     const uint16_t* PackedFilter,   // x1
//     float* Output,                  // x2
//     size_t OutputQuartetCount,      // x3
//     size_t InputChannels)           // x4
//
// PackedInput layout per output quartet:
//   input-channel major. Each input channel block contributes eight 8xBF16
//   records: k-groups 0-3 for rows 0-1 followed by k-groups 0-3 for rows 2-3.
//   This preserves the original 4-output accumulator shape while avoiding
//   repeated FP32-to-BF16 conversion for every filter tile.
//
// Register ownership:
//   x5    - remaining output quartets
//   x6    - packed input base for current output quartet
//   x7    - output base for current output quartet
//   x10   - packed input pointer within ic loop / store base
//   x11   - store pointer for row 1
//   x12   - packed filter pointer within ic loop
//   x13   - remaining input channels
//   x14/x15 - store pointers for rows 2-3
//
//   v16-v23 - accumulators for output rows 0-1
//   v24-v31 - accumulators for output rows 2-3
//   v8-v11  - staged packed BF16 inputs for rows 0-1
//   v12-v15 - staged packed BF16 inputs for rows 2-3
//   v0-v7   - filter loads and store staging
//

        FUNCTION_ENTRY MlasConvPointwiseBf16PackedInputKernelNeon4xAsm

    cbz     x3, 139f

    sub     sp, sp, #128
    stp     q8, q9, [sp, #0]
    stp     q10, q11, [sp, #32]
    stp     q12, q13, [sp, #64]
    stp     q14, q15, [sp, #96]

    mov     x5, x3              // remaining output quartets
    mov     x6, x0              // packed input base for current output quartet
    mov     x7, x2              // output base for current output quartet

120: // OutputQuartetCount loop
    eor     v16.16b, v16.16b, v16.16b
    eor     v17.16b, v17.16b, v17.16b
    eor     v18.16b, v18.16b, v18.16b
    eor     v19.16b, v19.16b, v19.16b
    eor     v20.16b, v20.16b, v20.16b
    eor     v21.16b, v21.16b, v21.16b
    eor     v22.16b, v22.16b, v22.16b
    eor     v23.16b, v23.16b, v23.16b
    eor     v24.16b, v24.16b, v24.16b
    eor     v25.16b, v25.16b, v25.16b
    eor     v26.16b, v26.16b, v26.16b
    eor     v27.16b, v27.16b, v27.16b
    eor     v28.16b, v28.16b, v28.16b
    eor     v29.16b, v29.16b, v29.16b
    eor     v30.16b, v30.16b, v30.16b
    eor     v31.16b, v31.16b, v31.16b

    mov     x10, x6             // packed input pointer
    mov     x12, x1             // packed filter pointer
    mov     x13, x4             // remaining input channels

121: // InputChannels loop
    prfm    pldl1keep, [x10, #256]
    prfm    pldl1keep, [x12, #512]

    ldp     q8, q9, [x10], #32
    ldp     q10, q11, [x10], #32
    ldp     q12, q13, [x10], #32
    ldp     q14, q15, [x10], #32

    ldp     q0, q1, [x12, #0]
    ldp     q2, q3, [x12, #32]
    ldp     q4, q5, [x12, #64]
    ldp     q6, q7, [x12, #96]

    bfmmla  v16.4s, v8.8h, v0.8h
    bfmmla  v17.4s, v8.8h, v1.8h
    bfmmla  v18.4s, v8.8h, v2.8h
    bfmmla  v19.4s, v8.8h, v3.8h
    bfmmla  v24.4s, v12.8h, v0.8h
    bfmmla  v25.4s, v12.8h, v1.8h
    bfmmla  v26.4s, v12.8h, v2.8h
    bfmmla  v27.4s, v12.8h, v3.8h

    ldp     q0, q1, [x12, #128]
    ldp     q2, q3, [x12, #160]

    bfmmla  v20.4s, v8.8h, v4.8h
    bfmmla  v21.4s, v8.8h, v5.8h
    bfmmla  v22.4s, v8.8h, v6.8h
    bfmmla  v23.4s, v8.8h, v7.8h
    bfmmla  v28.4s, v12.8h, v4.8h
    bfmmla  v29.4s, v12.8h, v5.8h
    bfmmla  v30.4s, v12.8h, v6.8h
    bfmmla  v31.4s, v12.8h, v7.8h

    ldp     q4, q5, [x12, #192]
    ldp     q6, q7, [x12, #224]

    bfmmla  v16.4s, v9.8h, v0.8h
    bfmmla  v17.4s, v9.8h, v1.8h
    bfmmla  v18.4s, v9.8h, v2.8h
    bfmmla  v19.4s, v9.8h, v3.8h
    bfmmla  v24.4s, v13.8h, v0.8h
    bfmmla  v25.4s, v13.8h, v1.8h
    bfmmla  v26.4s, v13.8h, v2.8h
    bfmmla  v27.4s, v13.8h, v3.8h

    ldp     q0, q1, [x12, #256]
    ldp     q2, q3, [x12, #288]

    bfmmla  v20.4s, v9.8h, v4.8h
    bfmmla  v21.4s, v9.8h, v5.8h
    bfmmla  v22.4s, v9.8h, v6.8h
    bfmmla  v23.4s, v9.8h, v7.8h
    bfmmla  v28.4s, v13.8h, v4.8h
    bfmmla  v29.4s, v13.8h, v5.8h
    bfmmla  v30.4s, v13.8h, v6.8h
    bfmmla  v31.4s, v13.8h, v7.8h

    ldp     q4, q5, [x12, #320]
    ldp     q6, q7, [x12, #352]

    bfmmla  v16.4s, v10.8h, v0.8h
    bfmmla  v17.4s, v10.8h, v1.8h
    bfmmla  v18.4s, v10.8h, v2.8h
    bfmmla  v19.4s, v10.8h, v3.8h
    bfmmla  v24.4s, v14.8h, v0.8h
    bfmmla  v25.4s, v14.8h, v1.8h
    bfmmla  v26.4s, v14.8h, v2.8h
    bfmmla  v27.4s, v14.8h, v3.8h

    ldp     q0, q1, [x12, #384]
    ldp     q2, q3, [x12, #416]

    bfmmla  v20.4s, v10.8h, v4.8h
    bfmmla  v21.4s, v10.8h, v5.8h
    bfmmla  v22.4s, v10.8h, v6.8h
    bfmmla  v23.4s, v10.8h, v7.8h
    bfmmla  v28.4s, v14.8h, v4.8h
    bfmmla  v29.4s, v14.8h, v5.8h
    bfmmla  v30.4s, v14.8h, v6.8h
    bfmmla  v31.4s, v14.8h, v7.8h

    ldp     q4, q5, [x12, #448]
    ldp     q6, q7, [x12, #480]

    bfmmla  v16.4s, v11.8h, v0.8h
    bfmmla  v17.4s, v11.8h, v1.8h
    bfmmla  v18.4s, v11.8h, v2.8h
    bfmmla  v19.4s, v11.8h, v3.8h
    bfmmla  v24.4s, v15.8h, v0.8h
    bfmmla  v25.4s, v15.8h, v1.8h
    bfmmla  v26.4s, v15.8h, v2.8h
    bfmmla  v27.4s, v15.8h, v3.8h
    bfmmla  v20.4s, v11.8h, v4.8h
    bfmmla  v21.4s, v11.8h, v5.8h
    bfmmla  v22.4s, v11.8h, v6.8h
    bfmmla  v23.4s, v11.8h, v7.8h
    bfmmla  v28.4s, v15.8h, v4.8h
    bfmmla  v29.4s, v15.8h, v5.8h
    bfmmla  v30.4s, v15.8h, v6.8h
    bfmmla  v31.4s, v15.8h, v7.8h

    add     x12, x12, #512
    subs    x13, x13, #1
    b.ne    121b

    mov     x6, x10
    mov     x10, x7
    add     x11, x10, #64

    zip1    v0.2d, v16.2d, v17.2d
    zip1    v1.2d, v18.2d, v19.2d
    zip1    v2.2d, v20.2d, v21.2d
    zip1    v3.2d, v22.2d, v23.2d
    zip2    v4.2d, v16.2d, v17.2d
    zip2    v5.2d, v18.2d, v19.2d
    zip2    v6.2d, v20.2d, v21.2d
    zip2    v7.2d, v22.2d, v23.2d

    stp     q0, q1, [x10], #32
    stp     q2, q3, [x10], #32
    stp     q4, q5, [x11], #32
    stp     q6, q7, [x11], #32

    add     x14, x7, #128
    add     x15, x7, #192

    zip1    v0.2d, v24.2d, v25.2d
    zip1    v1.2d, v26.2d, v27.2d
    zip1    v2.2d, v28.2d, v29.2d
    zip1    v3.2d, v30.2d, v31.2d
    zip2    v4.2d, v24.2d, v25.2d
    zip2    v5.2d, v26.2d, v27.2d
    zip2    v6.2d, v28.2d, v29.2d
    zip2    v7.2d, v30.2d, v31.2d

    stp     q0, q1, [x14], #32
    stp     q2, q3, [x14], #32
    stp     q4, q5, [x15], #32
    stp     q6, q7, [x15], #32

    add     x7, x7, #256
    subs    x5, x5, #1
    b.ne    120b

    ldp     q8, q9, [sp, #0]
    ldp     q10, q11, [sp, #32]
    ldp     q12, q13, [sp, #64]
    ldp     q14, q15, [sp, #96]
    add     sp, sp, #128
139:
    ret

//
// void MlasConvPointwiseBf16PackedInputKernelNeon2xAsm(
//     const uint16_t* PackedInput,   // x0
//     const uint16_t* PackedFilter,  // x1
//     float* Output,                 // x2
//     size_t OutputPairCount,        // x3
//     size_t InputChannels)          // x4
//
// PackedInput layout per output pair:
//   input-channel major. Each input channel block contributes four 8xBF16
//   records (k-groups 0-3). Each record stores the low row in lanes 0-3 and
//   the high row in lanes 4-7 so that the existing two-output BFMMLA flow can
//   be reused without repeating FP32-to-BF16 conversion for every filter tile.
//
// Register ownership:
//   x5    - remaining output pairs
//   x6    - packed input base for current output pair
//   x7    - output base for current output pair
//   x10   - packed input pointer within ic loop
//   x11   - packed filter pointer within ic loop
//   x12   - remaining input channels
//   x14/x15 - store pointers
//
//   v16-v23 - accumulators for the 8 column pairs
//   v8-v11  - staged packed BF16 inputs for k-groups 0-3
//   v0-v7   - filter loads and store staging
//

        FUNCTION_ENTRY MlasConvPointwiseBf16PackedInputKernelNeon2xAsm

    cbz     x3, 159f

    sub     sp, sp, #64
    stp     q8, q9, [sp, #0]
    stp     q10, q11, [sp, #32]

    mov     x5, x3              // remaining output pairs
    mov     x6, x0              // packed input base for current output pair
    mov     x7, x2              // output base for current output pair

140: // OutputPairCount loop
    eor     v16.16b, v16.16b, v16.16b
    eor     v17.16b, v17.16b, v17.16b
    eor     v18.16b, v18.16b, v18.16b
    eor     v19.16b, v19.16b, v19.16b
    eor     v20.16b, v20.16b, v20.16b
    eor     v21.16b, v21.16b, v21.16b
    eor     v22.16b, v22.16b, v22.16b
    eor     v23.16b, v23.16b, v23.16b

    mov     x10, x6             // packed input pointer
    mov     x11, x1             // packed filter pointer
    mov     x12, x4             // remaining input channels

141: // InputChannels loop
    prfm    pldl1keep, [x10, #256]
    prfm    pldl1keep, [x11, #512]

    ldp     q8, q9, [x10], #32
    ldp     q10, q11, [x10], #32

    ldp     q0, q1, [x11, #0]
    ldp     q2, q3, [x11, #32]
    ldp     q4, q5, [x11, #64]
    ldp     q6, q7, [x11, #96]

    bfmmla  v16.4s, v8.8h, v0.8h
    bfmmla  v17.4s, v8.8h, v1.8h
    bfmmla  v18.4s, v8.8h, v2.8h
    bfmmla  v19.4s, v8.8h, v3.8h

    ldp     q0, q1, [x11, #128]
    ldp     q2, q3, [x11, #160]

    bfmmla  v20.4s, v8.8h, v4.8h
    bfmmla  v21.4s, v8.8h, v5.8h
    bfmmla  v22.4s, v8.8h, v6.8h
    bfmmla  v23.4s, v8.8h, v7.8h

    ldp     q4, q5, [x11, #192]
    ldp     q6, q7, [x11, #224]

    bfmmla  v16.4s, v9.8h, v0.8h
    bfmmla  v17.4s, v9.8h, v1.8h
    bfmmla  v18.4s, v9.8h, v2.8h
    bfmmla  v19.4s, v9.8h, v3.8h

    ldp     q0, q1, [x11, #256]
    ldp     q2, q3, [x11, #288]

    bfmmla  v20.4s, v9.8h, v4.8h
    bfmmla  v21.4s, v9.8h, v5.8h
    bfmmla  v22.4s, v9.8h, v6.8h
    bfmmla  v23.4s, v9.8h, v7.8h

    ldp     q4, q5, [x11, #320]
    ldp     q6, q7, [x11, #352]

    bfmmla  v16.4s, v10.8h, v0.8h
    bfmmla  v17.4s, v10.8h, v1.8h
    bfmmla  v18.4s, v10.8h, v2.8h
    bfmmla  v19.4s, v10.8h, v3.8h

    ldp     q0, q1, [x11, #384]
    ldp     q2, q3, [x11, #416]

    bfmmla  v20.4s, v10.8h, v4.8h
    bfmmla  v21.4s, v10.8h, v5.8h
    bfmmla  v22.4s, v10.8h, v6.8h
    bfmmla  v23.4s, v10.8h, v7.8h

    ldp     q4, q5, [x11, #448]
    ldp     q6, q7, [x11, #480]

    bfmmla  v16.4s, v11.8h, v0.8h
    bfmmla  v17.4s, v11.8h, v1.8h
    bfmmla  v18.4s, v11.8h, v2.8h
    bfmmla  v19.4s, v11.8h, v3.8h
    bfmmla  v20.4s, v11.8h, v4.8h
    bfmmla  v21.4s, v11.8h, v5.8h
    bfmmla  v22.4s, v11.8h, v6.8h
    bfmmla  v23.4s, v11.8h, v7.8h

    add     x11, x11, #512
    subs    x12, x12, #1
    b.ne    141b

    mov     x6, x10
    mov     x14, x7
    add     x15, x14, #64

    zip1    v0.2d, v16.2d, v17.2d
    zip1    v1.2d, v18.2d, v19.2d
    zip1    v2.2d, v20.2d, v21.2d
    zip1    v3.2d, v22.2d, v23.2d
    zip2    v4.2d, v16.2d, v17.2d
    zip2    v5.2d, v18.2d, v19.2d
    zip2    v6.2d, v20.2d, v21.2d
    zip2    v7.2d, v22.2d, v23.2d

    stp     q0, q1, [x14], #32
    stp     q2, q3, [x14], #32
    stp     q4, q5, [x15], #32
    stp     q6, q7, [x15], #32

    add     x7, x7, #128
    subs    x5, x5, #1
    b.ne    140b

    ldp     q8, q9, [sp, #0]
    ldp     q10, q11, [sp, #32]
    add     sp, sp, #64
159:
    ret

//
// void MlasConvPointwiseBf16KernelNeonAsm(
//     const float* Input,            // x0
//     const uint16_t* PackedFilter,  // x1
//     float* Output,                 // x2
//     size_t StrideWidthBytes,       // x3
//     size_t InputStrideBytes,       // x4
//     size_t InputChannels,          // x5
//     size_t OutputCountEven)        // x6
//
// PackedFilter layout per input channel block:
//   k-group major (4 groups of 4 BF16 lanes) and, within each group,
//   column-pair major (8 pairs of output columns).
//   Each (k-group, column-pair) record is 8 BF16 values laid out as:
//     [b(k+0,c0), b(k+1,c0), b(k+2,c0), b(k+3,c0),
//      b(k+0,c1), b(k+1,c1), b(k+2,c1), b(k+3,c1)]
//
// Register ownership:
//   x7    - remaining output count (even)
//   x8    - input base pointer for current output index
//   x9    - output base pointer for current output index
//   x10   - input pointer (output row 0) within ic loop
//   x11   - input pointer (output row 1) within ic loop
//   x12   - packed filter pointer within ic loop
//   x13   - remaining input channels
//   x14   - input pointer (output row 2) within ic loop / scratch
//   x15   - input pointer (output row 3) within ic loop / scratch
//   x16   - StrideWidthBytes * 2
//   x17   - InputStrideBytes - (BlockSize * sizeof(float))
//   x6    - StrideWidthBytes * 4
//
//   v16-v23 - accumulators for output rows 0-1
//   v24-v31 - accumulators for output rows 2-3
//   v8-v11  - staged BF16 inputs for rows 0-1 (k-groups 0-3)
//   v12-v15 - staged BF16 inputs for rows 2-3 (k-groups 0-3)
//   v0-v3   - filter bank A / temporaries
//   v4-v7   - filter bank B / temporaries and store staging
//

        FUNCTION_ENTRY MlasConvPointwiseBf16KernelNeonAsm

    cbz     x6, 98f

    // The optimized hot path uses v8-v15 to stage all four k-groups of
    // converted BF16 input for a channel block. Preserve the callee-saved
    // registers once per call to keep the inner loop free of spills.
    sub     sp, sp, #128
    stp     q8, q9, [sp, #0]
    stp     q10, q11, [sp, #32]
    stp     q12, q13, [sp, #64]
    stp     q14, q15, [sp, #96]

    mov     x7, x6              // remaining outputs
    mov     x8, x0              // input base for current output index
    mov     x9, x2              // output base for current output index

    add     x16, x3, x3         // StrideWidthBytes * 2
    sub     x17, x4, #64        // InputStrideBytes - 64

    cmp     x7, #4
    b.lo    90f

    add     x6, x16, x16        // StrideWidthBytes * 4

80: // OutputCountEven loop (process 4 outputs per iteration)
    // Zero accumulators for rows 0-1.
    eor     v16.16b, v16.16b, v16.16b
    eor     v17.16b, v17.16b, v17.16b
    eor     v18.16b, v18.16b, v18.16b
    eor     v19.16b, v19.16b, v19.16b
    eor     v20.16b, v20.16b, v20.16b
    eor     v21.16b, v21.16b, v21.16b
    eor     v22.16b, v22.16b, v22.16b
    eor     v23.16b, v23.16b, v23.16b

    // Zero accumulators for rows 2-3.
    eor     v24.16b, v24.16b, v24.16b
    eor     v25.16b, v25.16b, v25.16b
    eor     v26.16b, v26.16b, v26.16b
    eor     v27.16b, v27.16b, v27.16b
    eor     v28.16b, v28.16b, v28.16b
    eor     v29.16b, v29.16b, v29.16b
    eor     v30.16b, v30.16b, v30.16b
    eor     v31.16b, v31.16b, v31.16b

    // Initialize per-iteration pointers.
    mov     x10, x8             // input row 0 base
    add     x11, x10, x3        // input row 1 base
    add     x14, x10, x16       // input row 2 base
    add     x15, x11, x16       // input row 3 base
    mov     x12, x1             // packed filter base
    mov     x13, x5             // remaining input channel blocks

81: // InputChannels loop (4 outputs)
    // Prefetch the next input/filter panels to keep the load pipelines fed.
    prfm    pldl1keep, [x10, #256]
    prfm    pldl1keep, [x12, #512]

    // Convert all four k-groups up front into v8-v15. Use LDP to fetch two
    // k-groups per row at a time, cutting the number of load instructions in
    // half without changing the dataflow seen by the BFMMLA core.
    // k-groups 0-1 input conversion (rows 0-1).
    ldp     q0, q1, [x10], #32
    ldp     q2, q3, [x11], #32
    bfcvtn  v8.4h, v0.4s
    bfcvtn2 v8.8h, v2.4s
    bfcvtn  v9.4h, v1.4s
    bfcvtn2 v9.8h, v3.4s

    // k-groups 2-3 input conversion (rows 0-1).
    ldp     q0, q1, [x10], #32
    ldp     q2, q3, [x11], #32
    bfcvtn  v10.4h, v0.4s
    bfcvtn2 v10.8h, v2.4s
    bfcvtn  v11.4h, v1.4s
    bfcvtn2 v11.8h, v3.4s

    // k-groups 0-1 input conversion (rows 2-3).
    ldp     q0, q1, [x14], #32
    ldp     q2, q3, [x15], #32
    bfcvtn  v12.4h, v0.4s
    bfcvtn2 v12.8h, v2.4s
    bfcvtn  v13.4h, v1.4s
    bfcvtn2 v13.8h, v3.4s

    // k-groups 2-3 input conversion (rows 2-3).
    ldp     q0, q1, [x14], #32
    ldp     q2, q3, [x15], #32
    bfcvtn  v14.4h, v0.4s
    bfcvtn2 v14.8h, v2.4s
    bfcvtn  v15.4h, v1.4s
    bfcvtn2 v15.8h, v3.4s

    // Advance to the next input channel block early; the pointers won't be
    // reused until the next iteration.
    add     x10, x10, x17
    add     x11, x11, x17
    add     x14, x14, x17
    add     x15, x15, x17

    // Preload k-group 0 filter panels into two register banks. This enables
    // loading the next k-group's column pairs 0-3 while column pairs 4-7 are
    // still accumulating, hiding most of the load-to-use latency.
    ldp     q0, q1, [x12, #0]
    ldp     q2, q3, [x12, #32]
    ldp     q4, q5, [x12, #64]
    ldp     q6, q7, [x12, #96]

    // k-group 0, column pairs 0-3.
    bfmmla  v16.4s, v8.8h, v0.8h
    bfmmla  v17.4s, v8.8h, v1.8h
    bfmmla  v18.4s, v8.8h, v2.8h
    bfmmla  v19.4s, v8.8h, v3.8h
    bfmmla  v24.4s, v12.8h, v0.8h
    bfmmla  v25.4s, v12.8h, v1.8h
    bfmmla  v26.4s, v12.8h, v2.8h
    bfmmla  v27.4s, v12.8h, v3.8h

    // Preload k-group 1, column pairs 0-3.
    ldp     q0, q1, [x12, #128]
    ldp     q2, q3, [x12, #160]

    // k-group 0, column pairs 4-7.
    bfmmla  v20.4s, v8.8h, v4.8h
    bfmmla  v21.4s, v8.8h, v5.8h
    bfmmla  v22.4s, v8.8h, v6.8h
    bfmmla  v23.4s, v8.8h, v7.8h
    bfmmla  v28.4s, v12.8h, v4.8h
    bfmmla  v29.4s, v12.8h, v5.8h
    bfmmla  v30.4s, v12.8h, v6.8h
    bfmmla  v31.4s, v12.8h, v7.8h

    // Preload k-group 1, column pairs 4-7.
    ldp     q4, q5, [x12, #192]
    ldp     q6, q7, [x12, #224]

    // k-group 1, column pairs 0-3.
    bfmmla  v16.4s, v9.8h, v0.8h
    bfmmla  v17.4s, v9.8h, v1.8h
    bfmmla  v18.4s, v9.8h, v2.8h
    bfmmla  v19.4s, v9.8h, v3.8h
    bfmmla  v24.4s, v13.8h, v0.8h
    bfmmla  v25.4s, v13.8h, v1.8h
    bfmmla  v26.4s, v13.8h, v2.8h
    bfmmla  v27.4s, v13.8h, v3.8h

    // Preload k-group 2, column pairs 0-3.
    ldp     q0, q1, [x12, #256]
    ldp     q2, q3, [x12, #288]

    // k-group 1, column pairs 4-7.
    bfmmla  v20.4s, v9.8h, v4.8h
    bfmmla  v21.4s, v9.8h, v5.8h
    bfmmla  v22.4s, v9.8h, v6.8h
    bfmmla  v23.4s, v9.8h, v7.8h
    bfmmla  v28.4s, v13.8h, v4.8h
    bfmmla  v29.4s, v13.8h, v5.8h
    bfmmla  v30.4s, v13.8h, v6.8h
    bfmmla  v31.4s, v13.8h, v7.8h

    // Preload k-group 2, column pairs 4-7.
    ldp     q4, q5, [x12, #320]
    ldp     q6, q7, [x12, #352]

    // k-group 2, column pairs 0-3.
    bfmmla  v16.4s, v10.8h, v0.8h
    bfmmla  v17.4s, v10.8h, v1.8h
    bfmmla  v18.4s, v10.8h, v2.8h
    bfmmla  v19.4s, v10.8h, v3.8h
    bfmmla  v24.4s, v14.8h, v0.8h
    bfmmla  v25.4s, v14.8h, v1.8h
    bfmmla  v26.4s, v14.8h, v2.8h
    bfmmla  v27.4s, v14.8h, v3.8h

    // Preload k-group 3, column pairs 0-3.
    ldp     q0, q1, [x12, #384]
    ldp     q2, q3, [x12, #416]

    // k-group 2, column pairs 4-7.
    bfmmla  v20.4s, v10.8h, v4.8h
    bfmmla  v21.4s, v10.8h, v5.8h
    bfmmla  v22.4s, v10.8h, v6.8h
    bfmmla  v23.4s, v10.8h, v7.8h
    bfmmla  v28.4s, v14.8h, v4.8h
    bfmmla  v29.4s, v14.8h, v5.8h
    bfmmla  v30.4s, v14.8h, v6.8h
    bfmmla  v31.4s, v14.8h, v7.8h

    // Preload k-group 3, column pairs 4-7.
    ldp     q4, q5, [x12, #448]
    ldp     q6, q7, [x12, #480]

    // k-group 3, column pairs 0-3.
    bfmmla  v16.4s, v11.8h, v0.8h
    bfmmla  v17.4s, v11.8h, v1.8h
    bfmmla  v18.4s, v11.8h, v2.8h
    bfmmla  v19.4s, v11.8h, v3.8h
    bfmmla  v24.4s, v15.8h, v0.8h
    bfmmla  v25.4s, v15.8h, v1.8h
    bfmmla  v26.4s, v15.8h, v2.8h
    bfmmla  v27.4s, v15.8h, v3.8h

    // k-group 3, column pairs 4-7.
    bfmmla  v20.4s, v11.8h, v4.8h
    bfmmla  v21.4s, v11.8h, v5.8h
    bfmmla  v22.4s, v11.8h, v6.8h
    bfmmla  v23.4s, v11.8h, v7.8h
    bfmmla  v28.4s, v15.8h, v4.8h
    bfmmla  v29.4s, v15.8h, v5.8h
    bfmmla  v30.4s, v15.8h, v6.8h
    bfmmla  v31.4s, v15.8h, v7.8h

    add     x12, x12, #512

    subs    x13, x13, #1
    b.ne    81b

    // Store rows 0-1: de-interleave via 64-bit ZIP.
    mov     x10, x9
    add     x11, x10, #64

    zip1    v0.2d, v16.2d, v17.2d
    zip1    v1.2d, v18.2d, v19.2d
    zip1    v2.2d, v20.2d, v21.2d
    zip1    v3.2d, v22.2d, v23.2d
    zip2    v4.2d, v16.2d, v17.2d
    zip2    v5.2d, v18.2d, v19.2d
    zip2    v6.2d, v20.2d, v21.2d
    zip2    v7.2d, v22.2d, v23.2d

    stp     q0, q1, [x10], #32
    stp     q2, q3, [x10], #32
    stp     q4, q5, [x11], #32
    stp     q6, q7, [x11], #32

    // Store rows 2-3.
    add     x14, x9, #128
    add     x15, x9, #192

    zip1    v0.2d, v24.2d, v25.2d
    zip1    v1.2d, v26.2d, v27.2d
    zip1    v2.2d, v28.2d, v29.2d
    zip1    v3.2d, v30.2d, v31.2d
    zip2    v4.2d, v24.2d, v25.2d
    zip2    v5.2d, v26.2d, v27.2d
    zip2    v6.2d, v28.2d, v29.2d
    zip2    v7.2d, v30.2d, v31.2d

    stp     q0, q1, [x14], #32
    stp     q2, q3, [x14], #32
    stp     q4, q5, [x15], #32
    stp     q6, q7, [x15], #32

    // Advance to the next block of outputs.
    subs    x7, x7, #4
    add     x8, x8, x6
    add     x9, x9, #256
    cmp     x7, #4
    b.hs    80b

90: // OutputCountEven loop (process 2 outputs per iteration)
    cbz     x7, 99f

    // Zero accumulators.
    eor     v16.16b, v16.16b, v16.16b
    eor     v17.16b, v17.16b, v17.16b
    eor     v18.16b, v18.16b, v18.16b
    eor     v19.16b, v19.16b, v19.16b
    eor     v20.16b, v20.16b, v20.16b
    eor     v21.16b, v21.16b, v21.16b
    eor     v22.16b, v22.16b, v22.16b
    eor     v23.16b, v23.16b, v23.16b

    // Initialize per-iteration pointers.
    mov     x10, x8             // input row 0 base
    add     x11, x10, x3        // input row 1 base
    mov     x12, x1             // packed filter base
    mov     x13, x5             // remaining input channel blocks

91: // InputChannels loop (2 outputs)
    // Prefetch the next panels.
    prfm    pldl1keep, [x10, #256]
    prfm    pldl1keep, [x12, #512]

    // Convert all four k-groups into v8-v11 using paired loads.
    ldp     q0, q1, [x10], #32
    ldp     q2, q3, [x11], #32
    bfcvtn  v8.4h, v0.4s
    bfcvtn2 v8.8h, v2.4s
    bfcvtn  v9.4h, v1.4s
    bfcvtn2 v9.8h, v3.4s

    ldp     q0, q1, [x10], #32
    ldp     q2, q3, [x11], #32
    bfcvtn  v10.4h, v0.4s
    bfcvtn2 v10.8h, v2.4s
    bfcvtn  v11.4h, v1.4s
    bfcvtn2 v11.8h, v3.4s

    // Advance to the next input channel block early.
    add     x10, x10, x17
    add     x11, x11, x17

    // Preload k-group 0 filter panels into two banks.
    ldp     q0, q1, [x12, #0]
    ldp     q2, q3, [x12, #32]
    ldp     q4, q5, [x12, #64]
    ldp     q6, q7, [x12, #96]

    // k-group 0, column pairs 0-3.
    bfmmla  v16.4s, v8.8h, v0.8h
    bfmmla  v17.4s, v8.8h, v1.8h
    bfmmla  v18.4s, v8.8h, v2.8h
    bfmmla  v19.4s, v8.8h, v3.8h

    // Preload k-group 1, column pairs 0-3.
    ldp     q0, q1, [x12, #128]
    ldp     q2, q3, [x12, #160]

    // k-group 0, column pairs 4-7.
    bfmmla  v20.4s, v8.8h, v4.8h
    bfmmla  v21.4s, v8.8h, v5.8h
    bfmmla  v22.4s, v8.8h, v6.8h
    bfmmla  v23.4s, v8.8h, v7.8h

    // Preload k-group 1, column pairs 4-7.
    ldp     q4, q5, [x12, #192]
    ldp     q6, q7, [x12, #224]

    // k-group 1, column pairs 0-3.
    bfmmla  v16.4s, v9.8h, v0.8h
    bfmmla  v17.4s, v9.8h, v1.8h
    bfmmla  v18.4s, v9.8h, v2.8h
    bfmmla  v19.4s, v9.8h, v3.8h

    // Preload k-group 2, column pairs 0-3.
    ldp     q0, q1, [x12, #256]
    ldp     q2, q3, [x12, #288]

    // k-group 1, column pairs 4-7.
    bfmmla  v20.4s, v9.8h, v4.8h
    bfmmla  v21.4s, v9.8h, v5.8h
    bfmmla  v22.4s, v9.8h, v6.8h
    bfmmla  v23.4s, v9.8h, v7.8h

    // Preload k-group 2, column pairs 4-7.
    ldp     q4, q5, [x12, #320]
    ldp     q6, q7, [x12, #352]

    // k-group 2, column pairs 0-3.
    bfmmla  v16.4s, v10.8h, v0.8h
    bfmmla  v17.4s, v10.8h, v1.8h
    bfmmla  v18.4s, v10.8h, v2.8h
    bfmmla  v19.4s, v10.8h, v3.8h

    // Preload k-group 3, column pairs 0-3.
    ldp     q0, q1, [x12, #384]
    ldp     q2, q3, [x12, #416]

    // k-group 2, column pairs 4-7.
    bfmmla  v20.4s, v10.8h, v4.8h
    bfmmla  v21.4s, v10.8h, v5.8h
    bfmmla  v22.4s, v10.8h, v6.8h
    bfmmla  v23.4s, v10.8h, v7.8h

    // Preload k-group 3, column pairs 4-7.
    ldp     q4, q5, [x12, #448]
    ldp     q6, q7, [x12, #480]

    // k-group 3, column pairs 0-3.
    bfmmla  v16.4s, v11.8h, v0.8h
    bfmmla  v17.4s, v11.8h, v1.8h
    bfmmla  v18.4s, v11.8h, v2.8h
    bfmmla  v19.4s, v11.8h, v3.8h

    // k-group 3, column pairs 4-7.
    bfmmla  v20.4s, v11.8h, v4.8h
    bfmmla  v21.4s, v11.8h, v5.8h
    bfmmla  v22.4s, v11.8h, v6.8h
    bfmmla  v23.4s, v11.8h, v7.8h

    add     x12, x12, #512

    // Advance to the next input channel block.
    subs    x13, x13, #1
    b.ne    91b

    // Store: de-interleave rows via 64-bit ZIP instructions.
    mov     x14, x9
    add     x15, x14, #64

    zip1    v0.2d, v16.2d, v17.2d
    zip1    v1.2d, v18.2d, v19.2d
    zip1    v2.2d, v20.2d, v21.2d
    zip1    v3.2d, v22.2d, v23.2d
    zip2    v4.2d, v16.2d, v17.2d
    zip2    v5.2d, v18.2d, v19.2d
    zip2    v6.2d, v20.2d, v21.2d
    zip2    v7.2d, v22.2d, v23.2d

    stp     q0, q1, [x14], #32
    stp     q2, q3, [x14], #32
    stp     q4, q5, [x15], #32
    stp     q6, q7, [x15], #32

    // Advance to the next pair of outputs.
    subs    x7, x7, #2
    add     x8, x8, x16
    add     x9, x9, #128
    b.ne    90b

99:
    ldp     q8, q9, [sp, #0]
    ldp     q10, q11, [sp, #32]
    ldp     q12, q13, [sp, #64]
    ldp     q14, q15, [sp, #96]
    add     sp, sp, #128
98:
    ret

//
// Single-output tail kernel.
//
// void MlasConvPointwiseBf16KernelNeonSingleOutputAsm(
//     const float* Input,            // x0
//     const uint16_t* PackedFilter,  // x1
//     float* Output,                 // x2
//     size_t StrideWidthBytes,       // x3
//     size_t InputStrideBytes,       // x4
//     size_t InputChannels,          // x5
//     size_t OutputCount)            // x6
//
// This kernel processes one output at a time and is used for odd-width
// remainders (or narrow outputs such as ow=1). It still uses BFMMLA by
// presenting the single output row as row 0 and a zero row as row 1.
//
// Register ownership:
//   x7    - remaining output count
//   x8    - input base pointer for current output index
//   x9    - output base pointer for current output index
//   x10   - input pointer within ic loop
//   x11   - packed filter pointer within ic loop
//   x12   - remaining input channels
//   x17   - InputStrideBytes - (BlockSize * sizeof(float))
//
//   v16-v23 - accumulators for the 8 column pairs
//   v0-v1   - FP32 input loads (two k-groups at a time)
//   v2/v7   - BF16 inputs for the active k-group (high half kept zero)
//   v3-v6   - filter loads and store staging
//

        FUNCTION_ENTRY MlasConvPointwiseBf16KernelNeonSingleOutputAsm

    cbz     x6, 199f

    mov     x7, x6              // remaining outputs
    mov     x8, x0              // input base for current output index
    mov     x9, x2              // output base for current output index

    sub     x17, x4, #64        // InputStrideBytes - 64

180: // OutputCount loop (process 1 output per iteration)
    // Zero accumulators.
    eor     v16.16b, v16.16b, v16.16b
    eor     v17.16b, v17.16b, v17.16b
    eor     v18.16b, v18.16b, v18.16b
    eor     v19.16b, v19.16b, v19.16b
    eor     v20.16b, v20.16b, v20.16b
    eor     v21.16b, v21.16b, v21.16b
    eor     v22.16b, v22.16b, v22.16b
    eor     v23.16b, v23.16b, v23.16b

    // Initialize per-iteration pointers.
    mov     x10, x8             // input row 0 base
    mov     x11, x1             // packed filter base
    mov     x12, x5             // remaining input channel blocks

181: // InputChannels loop (1 output)
    // Prefetch the next panels.
    prfm    pldl1keep, [x10, #256]
    prfm    pldl1keep, [x11, #512]

    // k-groups 0-1 input conversion (row 0 + implicit zero row 1).
    ldp     q0, q1, [x10], #32
    eor     v2.16b, v2.16b, v2.16b
    bfcvtn  v2.4h, v0.4s

    // k-group 0, column pairs 0-3.
    ld1     {v4.8h, v5.8h, v6.8h, v7.8h}, [x11], #64
    bfmmla  v16.4s, v2.8h, v4.8h
    bfmmla  v17.4s, v2.8h, v5.8h
    bfmmla  v18.4s, v2.8h, v6.8h
    bfmmla  v19.4s, v2.8h, v7.8h

    // k-group 0, column pairs 4-7.
    ld1     {v4.8h, v5.8h, v6.8h, v7.8h}, [x11], #64
    bfmmla  v20.4s, v2.8h, v4.8h
    bfmmla  v21.4s, v2.8h, v5.8h
    bfmmla  v22.4s, v2.8h, v6.8h
    bfmmla  v23.4s, v2.8h, v7.8h

    // k-group 1 (upper BF16 row is kept zero via the prior EOR).
    eor     v7.16b, v7.16b, v7.16b
    bfcvtn  v7.4h, v1.4s

    ld1     {v3.8h, v4.8h, v5.8h, v6.8h}, [x11], #64
    bfmmla  v16.4s, v7.8h, v3.8h
    bfmmla  v17.4s, v7.8h, v4.8h
    bfmmla  v18.4s, v7.8h, v5.8h
    bfmmla  v19.4s, v7.8h, v6.8h

    ld1     {v3.8h, v4.8h, v5.8h, v6.8h}, [x11], #64
    bfmmla  v20.4s, v7.8h, v3.8h
    bfmmla  v21.4s, v7.8h, v4.8h
    bfmmla  v22.4s, v7.8h, v5.8h
    bfmmla  v23.4s, v7.8h, v6.8h

    // k-groups 2-3 input conversion.
    ldp     q0, q1, [x10], #32
    eor     v2.16b, v2.16b, v2.16b
    bfcvtn  v2.4h, v0.4s

    ld1     {v3.8h, v4.8h, v5.8h, v6.8h}, [x11], #64
    bfmmla  v16.4s, v2.8h, v3.8h
    bfmmla  v17.4s, v2.8h, v4.8h
    bfmmla  v18.4s, v2.8h, v5.8h
    bfmmla  v19.4s, v2.8h, v6.8h

    ld1     {v3.8h, v4.8h, v5.8h, v6.8h}, [x11], #64
    bfmmla  v20.4s, v2.8h, v3.8h
    bfmmla  v21.4s, v2.8h, v4.8h
    bfmmla  v22.4s, v2.8h, v5.8h
    bfmmla  v23.4s, v2.8h, v6.8h

    eor     v7.16b, v7.16b, v7.16b
    bfcvtn  v7.4h, v1.4s

    ld1     {v3.8h, v4.8h, v5.8h, v6.8h}, [x11], #64
    bfmmla  v16.4s, v7.8h, v3.8h
    bfmmla  v17.4s, v7.8h, v4.8h
    bfmmla  v18.4s, v7.8h, v5.8h
    bfmmla  v19.4s, v7.8h, v6.8h

    ld1     {v3.8h, v4.8h, v5.8h, v6.8h}, [x11], #64
    bfmmla  v20.4s, v7.8h, v3.8h
    bfmmla  v21.4s, v7.8h, v4.8h
    bfmmla  v22.4s, v7.8h, v5.8h
    bfmmla  v23.4s, v7.8h, v6.8h

    // Advance to the next input channel block.
    add     x10, x10, x17
    subs    x12, x12, #1
    b.ne    181b

    // Store the single output row. Because the second row was zeroed, zip1
    // cleanly gathers the row-0 values without needing zip2.
    zip1    v0.2d, v16.2d, v17.2d
    zip1    v1.2d, v18.2d, v19.2d
    zip1    v2.2d, v20.2d, v21.2d
    zip1    v3.2d, v22.2d, v23.2d

    stp     q0, q1, [x9], #32
    stp     q2, q3, [x9], #32

    // Advance to the next output.
    add     x8, x8, x3
    subs    x7, x7, #1
    b.ne    180b

199:
    ret

#endif // __aarch64__ && __ARM_FEATURE_BF16
