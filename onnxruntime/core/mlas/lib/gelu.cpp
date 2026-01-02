/*++

Copyright 2025 FUJITSU LIMITED

Module Name:

   Gelu.cpp

Abstract:

    This module contains  Gelu helper functions .

--*/

#include "gelu.h"


void
MLASCALL
MlasComputeFP16Gelu(const MLAS_FP16* input,
                    MLAS_FP16* output,
                    MLAS_FP16* temp,
                    int64_t count,
                    const std::string& algo)
{
#if defined(MLAS_USE_SVE) || defined(MLAS_NEON_INTRINSICS)

    bool done = false;

#if defined(MLAS_USE_SVE)
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArmSve()) {
        MlasSveGeluF16Kernel(input, output, temp, count, algo);
        done = true;
    }
#endif

#if defined(MLAS_NEON_INTRINSICS)
    if (!done) {
        MlasNeonGeluF16Kernel(input, output, temp, count, algo);
        done = true;
    }
#endif

#else 

    (void)temp; 
    for (int64_t i = 0; i < count; ++i) {
        float x = static_cast<float>(input[i]);
        float gelu_val;

        if (algo == "tanh") {
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

#endif
}
