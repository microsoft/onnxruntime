/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    eltwise_kernel_neon_fp16.cpp

Abstract:

    This module implements the fp16 element-wise kernels for ARM NEON.

--*/
#include <arm_neon.h>
#include <cassert>

#include "fp16_common.h"
#include "eltwise.h"
#include "eltwise_kernel_neon.h"

namespace eltwise_neon {

void Add_Kernel_Fp16(const MLAS_FP16* left, const MLAS_FP16* right, MLAS_FP16* output, size_t N) {
    const auto* left_fp16 = reinterpret_cast<const _mlas_fp16_*>(left);
    const auto* right_fp16 = reinterpret_cast<const _mlas_fp16_*>(right);
    auto* output_fp16 = reinterpret_cast<_mlas_fp16_*>(output);

    while (N >= 32) {
        auto l0 = MlasLoadFloat16x8(left_fp16);
        auto l1 = MlasLoadFloat16x8(left_fp16 + 8);
        auto l2 = MlasLoadFloat16x8(left_fp16 + 16);
        auto l3 = MlasLoadFloat16x8(left_fp16 + 24);

        auto r0 = MlasLoadFloat16x8(right_fp16);
        auto r1 = MlasLoadFloat16x8(right_fp16 + 8);
        auto r2 = MlasLoadFloat16x8(right_fp16 + 16);
        auto r3 = MlasLoadFloat16x8(right_fp16 + 24);

        auto o0 = MlasAddFloat16(l0, r0);
        auto o1 = MlasAddFloat16(l1, r1);
        auto o2 = MlasAddFloat16(l2, r2);
        auto o3 = MlasAddFloat16(l3, r3);

        MlasStoreFloat16x8(output_fp16, o0);
        MlasStoreFloat16x8(output_fp16 + 8, o1);
        MlasStoreFloat16x8(output_fp16 + 16, o2);
        MlasStoreFloat16x8(output_fp16 + 24, o3);

        left_fp16 += 32;
        right_fp16 += 32;
        output_fp16 += 32;
        N -= 32;
    }

    if (N & 16) {
        auto l0 = MlasLoadFloat16x8(left_fp16);
        auto l1 = MlasLoadFloat16x8(left_fp16 + 8);

        auto r0 = MlasLoadFloat16x8(right_fp16);
        auto r1 = MlasLoadFloat16x8(right_fp16 + 8);

        auto o0 = MlasAddFloat16(l0, r0);
        auto o1 = MlasAddFloat16(l1, r1);

        MlasStoreFloat16x8(output_fp16, o0);
        MlasStoreFloat16x8(output_fp16 + 8, o1);

        left_fp16 += 16;
        right_fp16 += 16;
        output_fp16 += 16;
        N -= 16;
    }

    if (N & 8) {
        auto l0 = MlasLoadFloat16x8(left_fp16);
        auto r0 = MlasLoadFloat16x8(right_fp16);
        auto o0 = MlasAddFloat16(l0, r0);
        MlasStoreFloat16x8(output_fp16, o0);

        left_fp16 += 8;
        right_fp16 += 8;
        output_fp16 += 8;
        N -= 8;
    }

    if (N & 4) {
        auto l0 = MlasLoadFloat16x4(left_fp16);
        auto r0 = MlasLoadFloat16x4(right_fp16);
        auto o0 = MlasAddFloat16(l0, r0);
        MlasStoreFloat16x4(output_fp16, o0);

        left_fp16 += 4;
        right_fp16 += 4;
        output_fp16 += 4;
        N -= 4;
    }

    if (N == 3) {
        auto l0 = MlasLoadPartialFloat16x4(left_fp16, 3);
        auto r0 = MlasLoadPartialFloat16x4(right_fp16, 3);
        auto o0 = MlasAddFloat16(l0, r0);
        MlasStorePartialFloat16x4(output_fp16, o0, 3);
    } else if (N == 2) {
        auto l0 = MlasLoadPartialFloat16x4(left_fp16, 2);
        auto r0 = MlasLoadPartialFloat16x4(right_fp16, 2);
        auto o0 = MlasAddFloat16(l0, r0);
        MlasStorePartialFloat16x4(output_fp16, o0, 2);
    } else if (N == 1) {
        auto l0 = MlasLoadPartialFloat16x4(left_fp16, 1);
        auto r0 = MlasLoadPartialFloat16x4(right_fp16, 1);
        auto o0 = MlasAddFloat16(l0, r0);
        MlasStorePartialFloat16x4(output_fp16, o0, 1);
    }
}

}  // namespace eltwise_neon
