/*++

Copyright 2025 FUJITSU LIMITED
Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

   Gelu.cpp

Abstract:

    This module contains Gelu helper functions.

--*/
#include "mlasi.h"

void
MLASCALL
MlasComputeFP16Gelu(const MLAS_FP16* input,
                    MLAS_FP16* output,
                    MLAS_FP16* temp,
                    size_t count,
                    MLAS_GELU_ALGORITHM algo)
{
    if(GetMlasPlatform().GeluFP16KernelRoutine){
        GetMlasPlatform().GeluFP16KernelRoutine(input, output, temp, count, algo);
        return;
    }
    MLAS_UNREFERENCED_PARAMETER(temp); // 'temp' is only used by vectorized kernel implementations and it is unused in the scalar fallback path.
    for (size_t i = 0; i < count; ++i) {
        float x = static_cast<float>(input[i]);
        float gelu_val;

        if (algo == MlasGeluTanh) {
            // GELU approximation (tanh)
            const float B = 0.7978845608f;
            const float C = 0.044715f * B;
            float tanh_arg = x * (B + C * x * x);
            float tanh_res = std::tanh(tanh_arg);
            gelu_val = 0.5f * x * (1.0f + tanh_res);
        } else {
            // GELU exact (erf)
            gelu_val = 0.5f * x *
                (1.0f + std::erf(x * static_cast<float>(M_SQRT1_2)));
        }

        output[i] = MLAS_FP16(gelu_val);
    }
}
