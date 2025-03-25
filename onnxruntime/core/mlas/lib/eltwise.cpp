/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    eltwise.cpp

Abstract:

    This module implements routines to compute element-wise operations on two vectors.

    Currently supported element-wise operations:
        - Add

--*/

#include "mlasi.h"
#include "eltwise.h"

template <>
void
MLASCALL
MlasEltwiseAdd<float>(
    const float* left,
    const float* right,
    float* output,
    size_t N
) {
    while (N > 0) {
        if (N >= 4) {
            MLAS_FLOAT32X4 LeftVec = MlasLoadFloat32x4(left);
            MLAS_FLOAT32X4 RightVec = MlasLoadFloat32x4(right);

            MLAS_FLOAT32X4 ResultVec = MlasAddFloat32x4(LeftVec, RightVec);

            MlasStoreFloat32x4(output, ResultVec);

            left += 4;
            right += 4;
            output += 4;
            N -= 4;
        } else {
            *output = *left + *right;

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
    const auto* dispatch = GetMlasPlatform().EltwiseDispatch;
    if (dispatch == nullptr || dispatch->Add_Fp16 == nullptr) {
        MLAS_THROW_EX(std::runtime_error, "Add_Fp16 is not supported.");
    }
    dispatch->Add_Fp16(left, right, output, N);
}
