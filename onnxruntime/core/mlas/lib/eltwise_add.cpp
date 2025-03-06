/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    eltwise_add.cpp

Abstract:

    This module implements routines to compute eltwise addition of two vectors.

--*/

#include "mlasi.h"
#include <iostream>

template <>
void
MLASCALL
MlasEltwiseAdd<float>(
    const float* left,
    const float* right,
    float* output,
    size_t N
) {
    // std::cout << "running vectorized mlas kernel for fp32 addition" << std::endl;
    // for (size_t i = 0; i < N; i++) {
    //     output[i] = left[i] + right[i];
    // }

    while (N > 0) {
        MLAS_FLOAT32X4 LeftVec, RightVec;

        if (N >= 4) {
            LeftVec = MlasLoadFloat32x4(left);
            RightVec = MlasLoadFloat32x4(right);
        } else {
#if defined(MLAS_SSE2_INTRINSICS)
            // N.B. SSE2 lacks a broadcast load instruction, so avoid a shuffle
            // and use zeroes for the upper elements.
            LeftVec = _mm_load_ss(left);
            RightVec = _mm_load_ss(right);
#elif defined(MLAS_LSX_INTRINSICS)
            LeftVec = (MLAS_FLOAT32X4)__lsx_vldrepl_w(left, 0);
            RightVec = (MLAS_FLOAT32X4)__lsx_vldrepl_w(right, 0);
#else
            LeftVec = MlasBroadcastFloat32x4(left);
            RightVec = MlasBroadcastFloat32x4(right);
#endif
        }

        MLAS_FLOAT32X4 ResultVec = MlasAddFloat32x4(LeftVec, RightVec);

        if (N >= 4) {
            MlasStoreFloat32x4(output, ResultVec);

            left += 4;
            right += 4;
            output += 4;
            N -= 4;
        } else {
            MlasStoreLaneFloat32x4<0>(output, ResultVec);

            left += 1;
            right += 1;
            output += 1;
            N -= 1;
        }
    }
}


template <>
void
MLASCALL
MlasEltwiseAdd<MLAS_FP16>(
    const MLAS_FP16* left,
    const MLAS_FP16* right,
    MLAS_FP16* output,
    size_t N
) {
    MLAS_THROW_EX(std::runtime_error, "Mlas eltwise add for fp16 is not yet supported.");
}
